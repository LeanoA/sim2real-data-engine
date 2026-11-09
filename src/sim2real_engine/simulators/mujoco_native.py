"""Native DeepMind MuJoCo 4.x rigid-body physics and contact dynamics adapter."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from sim2real_engine.core.contracts import EpisodeStepRecord, JointState, SE3Pose
from sim2real_engine.models import load_robot_mjcf


class MuJoCoPhysicsAdapter:
    """Official MuJoCo 4.x C++ physics engine adapter for real contact dynamics.

    Implements the hexagonal SimulatorBackend port using compiled MJCF models,
    true Coulomb friction, rigid-body contact manifolds, and analytical C++ Jacobians.
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
        self._episode_id = 0
        self._step_index = 0

    @property
    def dof(self) -> int:
        """Number of articulated manipulator degrees of freedom."""
        return self._dof

    @property
    def home_configuration(self) -> NDArray[np.float64]:
        """Nominal home joint configuration in radians."""
        return self._home_q.copy()

    def reset(self, seed: int = 42) -> tuple[JointState, SE3Pose, SE3Pose]:
        """Reset physics simulation and deterministically perturb the payload position."""
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        self._episode_id = seed
        self._step_index = 0

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
        """Advance physical simulation through multiple substeps with collision solving."""
        if len(joint_targets_rad) != self._dof:
            msg = f"Expected {self._dof} joint targets; got {len(joint_targets_rad)}"
            raise ValueError(msg)

        self.data.ctrl[:6] = np.asarray(joint_targets_rad, dtype=np.float64)

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
        )
