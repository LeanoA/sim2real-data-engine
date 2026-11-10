"""Simulator backend protocol and privileged pick-and-place trajectory expert."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from sim2real_engine.core.contracts import EpisodeStepRecord, JointState, SE3Pose
from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.randomization.samplers import EpisodePerturbation

ExpertStage = Literal["APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "COMPLETED"]

# Maximum TCP-to-payload distance at which the gripper can latch the payload.
# Cube half-size is 20 mm; the privileged expert latches at 23-33 mm. Beyond this radius a
# close command is a missed grasp: the site-to-site weld would otherwise snap the payload
# to the TCP from any distance.
GRASP_ENGAGE_RADIUS_M = 0.04


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

    def reset(
        self,
        seed: int = 42,
        perturbation: EpisodePerturbation | None = None,
    ) -> tuple[JointState, SE3Pose, SE3Pose]:
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

    @property
    def gripper_state(self) -> float:
        """Actual gripper latch state: 1.0 when held, else 0.0."""
        ...

    def set_gripper(self, grasped: bool) -> bool:
        """Control gripper grasp or weld constraint attachment state."""
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
        place_target_pose: SE3Pose | None = None,
    ) -> None:
        self.ik_solver = ik_solver
        self.hover_offset_m = hover_offset_m
        self.lift_offset_m = lift_offset_m
        self.waypoint_tol_m = waypoint_tol_m
        self.max_step_rad = max_step_rad
        self.place_target_pose = place_target_pose
        self._stage: ExpertStage = "APPROACH"
        self._grasp_dwell_ticks: int = 0
        self._initial_obj_pose: SE3Pose = SE3Pose()
        self._active_place_target: SE3Pose = SE3Pose()

    @property
    def stage(self) -> ExpertStage:
        """Current finite-state machine stage."""
        return self._stage

    def reset(
        self,
        initial_object_pose: SE3Pose,
        place_target_pose: SE3Pose | None = None,
    ) -> None:
        """Reset expert state machine for a new episode."""
        self._stage = "APPROACH"
        self._grasp_dwell_ticks = 0
        self._initial_obj_pose = initial_object_pose
        target = place_target_pose or self.place_target_pose
        if target is not None:
            self._active_place_target = target
        else:
            # Default delivery zone on workstation table surface (y = -0.10)
            quat = initial_object_pose.quaternion_wxyz
            self._active_place_target = SE3Pose(
                position=(0.38, -0.10, initial_object_pose.position[2]),
                quaternion_wxyz=quat,
            )

    def current_waypoint(self) -> SE3Pose:
        """Return the active SE(3) waypoint for the current state machine stage."""
        ox, oy, oz = self._initial_obj_pose.position
        quat = self._initial_obj_pose.quaternion_wxyz
        px, py, pz = self._active_place_target.position
        p_quat = self._active_place_target.quaternion_wxyz

        if self._stage == "APPROACH":
            return SE3Pose(position=(ox, oy, oz + self.hover_offset_m), quaternion_wxyz=quat)
        if self._stage in ("DESCEND", "GRASP"):
            return SE3Pose(position=(ox, oy, oz), quaternion_wxyz=quat)
        if self._stage == "LIFT":
            return SE3Pose(position=(ox, oy, oz + self.lift_offset_m), quaternion_wxyz=quat)
        if self._stage == "TRANSPORT":
            return SE3Pose(position=(px, py, pz + self.lift_offset_m), quaternion_wxyz=p_quat)
        # PLACE or COMPLETED
        return SE3Pose(position=(px, py, pz), quaternion_wxyz=p_quat)

    def compute_action(
        self,
        q_current: NDArray[np.float64],
        q_home: NDArray[np.float64],
        ee_pose: SE3Pose,
        jacobian: NDArray[np.float64],
        emit_gripper: bool = True,
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
            self._stage = "TRANSPORT"
            waypoint = self.current_waypoint()
        elif self._stage == "TRANSPORT" and pos_dist <= self.waypoint_tol_m:
            self._stage = "PLACE"
            waypoint = self.current_waypoint()
        elif self._stage == "PLACE" and pos_dist <= min(self.waypoint_tol_m, 0.01):
            self._stage = "COMPLETED"

        delta_q = self.ik_solver.step(
            q_current=q_current,
            q_home=q_home,
            current_pose=ee_pose,
            target_pose=waypoint,
            jacobian=jacobian,
            max_step_rad=self.max_step_rad,
        )
        q_target = tuple(float(val) for val in (q_current + delta_q))

        if not emit_gripper:
            return q_target

        if self._stage in ("LIFT", "TRANSPORT", "PLACE"):
            gripper_cmd = 1.0
        elif self._stage in ("APPROACH", "DESCEND", "COMPLETED"):
            gripper_cmd = 0.0
        elif self._stage == "GRASP":
            gripper_cmd = 1.0 if self._grasp_dwell_ticks >= 3 else 0.0
        else:
            gripper_cmd = 0.0

        return (*q_target, gripper_cmd)

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

        for step_idx in range(max_steps):
            q_cur = np.asarray(joint_state.positions_rad, dtype=np.float64)
            jacobian = backend.get_jacobian()
            action_targets = self.compute_action(
                q_current=q_cur,
                q_home=q_home,
                ee_pose=ee_pose,
                jacobian=jacobian,
                emit_gripper=True,
            )

            # Store pre-step observation (s_t) with action (a_t)
            pre_joint_state = joint_state
            pre_ee_pose = ee_pose
            pre_obj_pose = obj_pose
            pre_gripper_state = (
                backend.gripper_state if hasattr(backend, "gripper_state") else 0.0
            )

            step_record = backend.step(action_targets)
            aligned_record = EpisodeStepRecord(
                episode_id=step_record.episode_id,
                step_index=step_idx,
                timestamp_s=round(step_idx * 0.02, 6),
                joint_state=pre_joint_state,
                end_effector_pose=pre_ee_pose,
                privileged_object_pose=pre_obj_pose,
                action_joint_targets_rad=action_targets,
                is_terminal=False,
                gripper_state=pre_gripper_state,
            )

            if self._stage == "COMPLETED":
                records.append(aligned_record)
                # Detach object and allow physical settling ticks
                release_action = (*action_targets[: len(q_home)], 0.0)
                for settle_idx in range(5):
                    settle_pre_joints = step_record.joint_state
                    settle_pre_ee = step_record.end_effector_pose
                    settle_pre_obj = step_record.privileged_object_pose
                    settle_pre_grip = (
                        backend.gripper_state if hasattr(backend, "gripper_state") else 0.0
                    )
                    step_record = backend.step(release_action)
                    curr_step = step_idx + 1 + settle_idx
                    settle_rec = EpisodeStepRecord(
                        episode_id=step_record.episode_id,
                        step_index=curr_step,
                        timestamp_s=round(curr_step * 0.02, 6),
                        joint_state=settle_pre_joints,
                        end_effector_pose=settle_pre_ee,
                        privileged_object_pose=settle_pre_obj,
                        action_joint_targets_rad=release_action,
                        is_terminal=(settle_idx == 4),
                        gripper_state=settle_pre_grip,
                    )
                    records.append(settle_rec)
                break

            records.append(aligned_record)
            joint_state = step_record.joint_state
            ee_pose = step_record.end_effector_pose
            obj_pose = step_record.privileged_object_pose

        return records
