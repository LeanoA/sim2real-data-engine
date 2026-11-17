"""Native DeepMind MuJoCo 4.x rigid-body physics and contact dynamics adapter."""

from __future__ import annotations

import math

import mujoco
import numpy as np
from numpy.typing import NDArray

from sim2real_engine.core.contracts import EpisodeStepRecord, JointState, SE3Pose
from sim2real_engine.models import load_robot_mjcf
from sim2real_engine.randomization.samplers import EpisodePerturbation


class MuJoCoPhysicsAdapter:
    """Official MuJoCo 4.x C++ physics engine adapter for real contact dynamics.

    Implements the hexagonal SimulatorBackend port using compiled MJCF models,
    articulated parallel-jaw fingers, true Coulomb friction, rigid-body contact
    manifolds, and analytical C++ Jacobians.
    """

    def __init__(self, control_dt: float = 0.02, physics_substeps: int = 10) -> None:
        self._control_dt = control_dt
        self._physics_substeps = physics_substeps
        self._dof = 6

        self.model, self.data = load_robot_mjcf()
        self.model.opt.timestep = control_dt / physics_substeps

        # Named entity identifiers conforming to modern MuJoCo Python conventions
        self._ee_body_id = self.model.body("end_effector").id
        self._home_key_id = self.model.key("home").id
        self._home_q = np.array([0.0, -0.55, 1.10, 0.0, -0.55, 0.0], dtype=np.float64)
        self._finger_travel_max: float = float(self.model.joint("finger_left_joint").range[1])
        self._gripper_ctrl_max: float = float(self.model.actuator("act_gripper").ctrlrange[1])
        self._gripper_target: float = 0.0
        self._episode_id = 0
        self._step_index = 0

        # Cache nominal physical parameters for domain randomization
        self._nominal_payload_mass: float = float(self.model.body("payload_object").mass[0])
        self._nominal_friction: tuple[float, float, float] = (
            float(self.model.geom("payload_geom").friction[0]),
            float(self.model.geom("payload_geom").friction[1]),
            float(self.model.geom("payload_geom").friction[2]),
        )
        self._nominal_finger_left_friction: tuple[float, float, float] = (
            float(self.model.geom("finger_left").friction[0]),
            float(self.model.geom("finger_left").friction[1]),
            float(self.model.geom("finger_left").friction[2]),
        )
        self._nominal_finger_right_friction: tuple[float, float, float] = (
            float(self.model.geom("finger_right").friction[0]),
            float(self.model.geom("finger_right").friction[1]),
            float(self.model.geom("finger_right").friction[2]),
        )
        self._nominal_dof_damping: NDArray[np.float64] = self.model.dof_damping[:6].copy()

    @property
    def dof(self) -> int:
        """Number of articulated manipulator degrees of freedom."""
        return self._dof

    @property
    def home_configuration(self) -> NDArray[np.float64]:
        """Nominal home joint configuration in radians."""
        return self._home_q.copy()

    def apply_perturbation(self, perturbation: EpisodePerturbation) -> None:
        """Apply physical parameter scaling (mass, friction, damping) to the MuJoCo model."""
        self.model.body("payload_object").mass = [
            self._nominal_payload_mass * perturbation.mass_scale
        ]
        self.model.geom("payload_geom").friction = [
            self._nominal_friction[0] * perturbation.friction_scale,
            self._nominal_friction[1] * perturbation.friction_scale,
            self._nominal_friction[2] * perturbation.friction_scale,
        ]
        self.model.geom("finger_left").friction = [
            self._nominal_finger_left_friction[0] * perturbation.friction_scale,
            self._nominal_finger_left_friction[1] * perturbation.friction_scale,
            self._nominal_finger_left_friction[2] * perturbation.friction_scale,
        ]
        self.model.geom("finger_right").friction = [
            self._nominal_finger_right_friction[0] * perturbation.friction_scale,
            self._nominal_finger_right_friction[1] * perturbation.friction_scale,
            self._nominal_finger_right_friction[2] * perturbation.friction_scale,
        ]
        self.model.dof_damping[:6] = self._nominal_dof_damping * perturbation.damping_scale
        mujoco.mj_setConst(self.model, self.data)

    def restore_nominal_physics(self) -> None:
        """Restore nominal payload mass, friction, and joint damping."""
        self.model.body("payload_object").mass = [self._nominal_payload_mass]
        self.model.geom("payload_geom").friction = list(self._nominal_friction)
        self.model.geom("finger_left").friction = list(self._nominal_finger_left_friction)
        self.model.geom("finger_right").friction = list(self._nominal_finger_right_friction)
        self.model.dof_damping[:6] = self._nominal_dof_damping.copy()
        mujoco.mj_setConst(self.model, self.data)

    def reset(
        self,
        seed: int = 42,
        perturbation: EpisodePerturbation | None = None,
    ) -> tuple[JointState, SE3Pose, SE3Pose]:
        """Reset physics simulation and deterministically perturb the payload position."""
        if perturbation is not None:
            self.apply_perturbation(perturbation)
        else:
            self.restore_nominal_physics()

        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        self._episode_id = seed
        self._step_index = 0
        self._gripper_target = 0.0
        self.data.actuator("act_gripper").ctrl[0] = 0.0

        # Perturb object on table surface (tabletop is at z = 0.05, box half-size is 0.02)
        obj_x = 0.38 + float(rng.uniform(-0.04, 0.04))
        obj_y = 0.05 + float(rng.uniform(-0.05, 0.05))
        obj_z = 0.07

        qpos_obj = np.array([obj_x, obj_y, obj_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.joint("payload_freejoint").qpos[:] = qpos_obj

        # Compute forward kinematics & contact state
        mujoco.mj_forward(self.model, self.data)

        ee_pose = self.get_current_ee_pose()
        obj_pose = self.get_target_pose()

        joint_state = JointState(
            positions_rad=tuple(float(v) for v in self.data.qpos[:6]),
            velocities_rad_s=tuple(float(v) for v in self.data.qvel[:6]),
            torques_nm=tuple(float(v) for v in self.data.actuator_force[:6]),
        )
        return joint_state, ee_pose, obj_pose

    def get_current_ee_pose(self) -> SE3Pose:
        """Return current end-effector SE(3) pose from MuJoCo kinematics."""
        ee = self.data.body("end_effector")
        pos = ee.xpos
        quat = ee.xquat
        return SE3Pose.from_unnormalized(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            quaternion_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        )

    def get_target_pose(self) -> SE3Pose:
        """Return current privileged payload object SE(3) pose."""
        obj = self.data.body("payload_object")
        pos = obj.xpos
        quat = obj.xquat
        return SE3Pose.from_unnormalized(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            quaternion_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        )

    def tcp_to_payload_distance(self) -> float:
        """Return the Euclidean distance between the gripper TCP and the payload site."""
        tcp = self.data.site("ee_tcp").xpos
        payload = self.data.site("payload_site").xpos
        return float(np.linalg.norm(tcp - payload))

    @property
    def gripper_state(self) -> float:
        """Normalized parallel-jaw closure in [0.0, 1.0] (0.0 = open, 1.0 = fully closed)."""
        q_left = float(self.data.joint("finger_left_joint").qpos[0])
        q_right = float(self.data.joint("finger_right_joint").qpos[0])
        avg_travel = 0.5 * (q_left + q_right)
        return float(np.clip(avg_travel / self._finger_travel_max, 0.0, 1.0))

    @property
    def gripper_contact_force(self) -> float:
        """Total scalar normal contact force (N) across left and right fingertip touch sensors."""
        left = float(self.data.sensor("touch_left").data[0])
        right = float(self.data.sensor("touch_right").data[0])
        return left + right

    def set_gripper(self, grasped: bool | float) -> bool:
        """Command the articulated parallel-jaw actuator target.

        Accepts a boolean (True=1.0 closed, False=0.0 open) or continuous normalized closure
        command in [0.0, 1.0]. Returns True when closure target > 0.5.
        """
        target = (1.0 if grasped else 0.0) if isinstance(grasped, bool) else float(grasped)
        if not math.isfinite(target):
            msg = f"Gripper closure target must be finite; got {target}"
            raise ValueError(msg)
        self._gripper_target = float(np.clip(target, 0.0, 1.0))
        self.data.actuator("act_gripper").ctrl[0] = self._gripper_target * self._gripper_ctrl_max
        return self._gripper_target > 0.5

    def get_jacobian(self) -> NDArray[np.float64]:
        """Compute exact 6x6 spatial geometric Jacobian at the end-effector via MuJoCo C API."""
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        ee_pos = self.data.body("end_effector").xpos

        mujoco.mj_jac(self.model, self.data, jacp, jacr, ee_pos, self._ee_body_id)

        # Extract first 6 degrees of freedom (manipulator joints)
        jac = np.zeros((6, 6), dtype=np.float64)
        jac[:3, :] = jacp[:, :6]
        jac[3:, :] = jacr[:, :6]
        return jac

    def step(self, joint_targets_rad: tuple[float, ...]) -> EpisodeStepRecord:
        """Advance physical simulation through multiple substeps with collision solving.

        Supports 6-DOF arm targets and 7-DOF reactive action targets (arm + gripper).
        When 7 targets are provided, index 6 commands normalized gripper closure in [0.0, 1.0].
        """
        if len(joint_targets_rad) == self._dof:
            arm_targets = joint_targets_rad
        elif len(joint_targets_rad) == self._dof + 1:
            arm_targets = joint_targets_rad[: self._dof]
            gripper_cmd = float(joint_targets_rad[self._dof])
            self.set_gripper(gripper_cmd)
        else:
            msg = (
                f"Expected {self._dof} (arm only) or {self._dof + 1} (arm + gripper) "
                f"action targets; got {len(joint_targets_rad)}"
            )
            raise ValueError(msg)

        self.data.ctrl[:6] = np.asarray(arm_targets, dtype=np.float64)
        self.data.actuator("act_gripper").ctrl[0] = self._gripper_target * self._gripper_ctrl_max

        for _ in range(self._physics_substeps):
            # Feedforward gravity compensation for high-precision articulated trajectory tracking
            self.data.qfrc_applied[:6] = self.data.qfrc_bias[:6]
            mujoco.mj_step(self.model, self.data)

        self._step_index += 1

        ee_pose = self.get_current_ee_pose()
        obj_pose = self.get_target_pose()

        joint_state = JointState(
            positions_rad=tuple(float(v) for v in self.data.qpos[:6]),
            velocities_rad_s=tuple(float(v) for v in self.data.qvel[:6]),
            torques_nm=tuple(float(v) for v in self.data.actuator_force[:6]),
        )

        return EpisodeStepRecord(
            episode_id=self._episode_id,
            step_index=self._step_index,
            timestamp_s=round(self._step_index * self._control_dt, 6),
            joint_state=joint_state,
            end_effector_pose=ee_pose,
            privileged_object_pose=obj_pose,
            action_joint_targets_rad=joint_targets_rad,
            is_terminal=False,
            gripper_state=self.gripper_state,
        )
