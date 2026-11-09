"""Simulator backend protocol and privileged pick-and-place trajectory expert."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from sim2real_engine.core.contracts import EpisodeStepRecord, JointState, SE3Pose
from sim2real_engine.core.kinematics import DampedLeastSquaresIK

ExpertStage = Literal["APPROACH", "DESCEND", "GRASP", "LIFT", "COMPLETED"]


class RolloutSummary(BaseModel):
    """Summary metrics for a batch of privileged expert episode rollouts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    num_episodes: int
    success_count: int
    success_rate: float
    mean_steps: float
    mean_final_ee_error_m: float


@runtime_checkable
class SimulatorBackend(Protocol):
    """Hexagonal port interface for physics simulation backends."""

    @property
    def dof(self) -> int:
        """Number of articulated degrees of freedom."""
        ...

    @property
    def home_configuration(self) -> NDArray[np.float64]:
        """Nominal home joint configuration in radians."""
        ...

    def reset(self, seed: int = 42) -> tuple[JointState, SE3Pose, SE3Pose]:
        """Reset episode state with a deterministic seed; returns (joints, ee_pose, object_pose)."""
        ...

    def step(self, joint_targets_rad: tuple[float, ...]) -> EpisodeStepRecord:
        """Advance physics by one control step given target joint positions."""
        ...

    def get_jacobian(self) -> NDArray[np.float64]:
        """Compute 6xN geometric Jacobian at the current joint configuration."""
        ...

    def get_current_ee_pose(self) -> SE3Pose:
        """Return current end-effector SE(3) pose in world coordinates."""
        ...

    def get_target_pose(self) -> SE3Pose:
        """Return current privileged object SE(3) pose in world coordinates."""
        ...


class PrivilegedPickPlaceExpert:
    """Closed-loop state machine expert using Damped Least-Squares IK for pick-and-place."""

    def __init__(
        self,
        ik_solver: DampedLeastSquaresIK,
        hover_offset_m: float = 0.10,
        lift_offset_m: float = 0.12,
        waypoint_tol_m: float = 0.035,
        max_step_rad: float = 0.12,
    ) -> None:
        self.ik_solver = ik_solver
        self.hover_offset_m = hover_offset_m
        self.lift_offset_m = lift_offset_m
        self.waypoint_tol_m = waypoint_tol_m
        self.max_step_rad = max_step_rad
        self._stage: ExpertStage = "APPROACH"
        self._grasp_dwell_ticks: int = 0
        self._initial_obj_pose: SE3Pose = SE3Pose()

    @property
    def stage(self) -> ExpertStage:
        """Current finite-state machine stage."""
        return self._stage

    def reset(self, initial_object_pose: SE3Pose) -> None:
        """Reset expert state machine for a new episode."""
        self._stage = "APPROACH"
        self._grasp_dwell_ticks = 0
        self._initial_obj_pose = initial_object_pose

    def current_waypoint(self) -> SE3Pose:
        """Return the active SE(3) waypoint for the current state machine stage."""
        ox, oy, oz = self._initial_obj_pose.position
        quat = self._initial_obj_pose.quaternion_wxyz
        if self._stage == "APPROACH":
            return SE3Pose(position=(ox, oy, oz + self.hover_offset_m), quaternion_wxyz=quat)
        if self._stage in ("DESCEND", "GRASP"):
            return SE3Pose(position=(ox, oy, oz), quaternion_wxyz=quat)
        return SE3Pose(position=(ox, oy, oz + self.lift_offset_m), quaternion_wxyz=quat)

    def compute_action(
        self,
        q_current: NDArray[np.float64],
        q_home: NDArray[np.float64],
        ee_pose: SE3Pose,
        jacobian: NDArray[np.float64],
    ) -> tuple[float, ...]:
        """Compute next joint position targets and advance the expert state machine."""
        waypoint = self.current_waypoint()
        pos_dist = float(
            np.linalg.norm(
                np.asarray(ee_pose.position, dtype=np.float64)
                - np.asarray(waypoint.position, dtype=np.float64)
            )
        )

        if self._stage == "APPROACH" and pos_dist <= self.waypoint_tol_m:
            self._stage = "DESCEND"
            waypoint = self.current_waypoint()
        elif self._stage == "DESCEND" and pos_dist <= self.waypoint_tol_m:
            self._stage = "GRASP"
            self._grasp_dwell_ticks = 0
            waypoint = self.current_waypoint()
        elif self._stage == "GRASP":
            self._grasp_dwell_ticks += 1
            if self._grasp_dwell_ticks >= 3:
                self._stage = "LIFT"
                waypoint = self.current_waypoint()
        elif self._stage == "LIFT" and pos_dist <= self.waypoint_tol_m:
            self._stage = "COMPLETED"

        delta_q = self.ik_solver.step(
            q_current=q_current,
            q_home=q_home,
            current_pose=ee_pose,
            target_pose=waypoint,
            jacobian=jacobian,
            max_step_rad=self.max_step_rad,
        )
        q_target = q_current + delta_q
        return tuple(float(val) for val in q_target)

    def run_episode(
        self,
        backend: SimulatorBackend,
        seed: int = 42,
        max_steps: int = 120,
    ) -> list[EpisodeStepRecord]:
        """Execute a complete closed-loop pick-and-place demonstration rollout."""
        joint_state, ee_pose, obj_pose = backend.reset(seed=seed)
        self.reset(initial_object_pose=obj_pose)
        q_home = backend.home_configuration
        records: list[EpisodeStepRecord] = []

        for _ in range(max_steps):
            q_cur = np.asarray(joint_state.positions_rad, dtype=np.float64)
            jacobian = backend.get_jacobian()
            action_targets = self.compute_action(
                q_current=q_cur,
                q_home=q_home,
                ee_pose=ee_pose,
                jacobian=jacobian,
            )
            step_record = backend.step(action_targets)
            if self._stage == "COMPLETED":
                step_record = step_record.model_copy(update={"is_terminal": True})
                records.append(step_record)
                break
            records.append(step_record)
            joint_state = step_record.joint_state
            ee_pose = step_record.end_effector_pose

        return records
