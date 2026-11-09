"""Local Apple Silicon ARM64 / CPU deterministic 6-DOF manipulator physics adapter."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from sim2real_engine.core.contracts import EpisodeStepRecord, JointState, SE3Pose
from sim2real_engine.core.kinematics import so3_log_map


def _rot_x(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_matrix_to_quaternion_wxyz(
    rot: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Convert a 3x3 proper rotation matrix to a unit quaternion (qw, qx, qy, qz)."""
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot[2, 1] - rot[1, 2]) * s
        qy = (rot[0, 2] - rot[2, 0]) * s
        qz = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return (float(qw / norm), float(qx / norm), float(qy / norm), float(qz / norm))


class MuJoCoKinematicAdapter:
    """Deterministic 6-DOF articulated arm simulator for local ARM64/CPU verification."""

    def __init__(
        self,
        control_dt: float = 0.02,
        kp: float = 45.0,
        kd: float = 4.0,
        tracking_alpha: float = 0.85,
    ) -> None:
        self._dof = 6
        self._control_dt = control_dt
        self._kp = kp
        self._kd = kd
        self._tracking_alpha = tracking_alpha
        self._home_q = np.array([0.0, -0.55, 1.10, 0.0, -0.55, 0.0], dtype=np.float64)
        self._q = self._home_q.copy()
        self._dq = np.zeros(self._dof, dtype=np.float64)
        self._torques = np.zeros(self._dof, dtype=np.float64)
        self._episode_id = 0
        self._step_index = 0
        self._object_grasped = False
        self._object_pose = SE3Pose(position=(0.38, 0.05, 0.12))

    @property
    def dof(self) -> int:
        """Number of articulated degrees of freedom."""
        return self._dof

    @property
    def home_configuration(self) -> NDArray[np.float64]:
        """Nominal home joint configuration in radians."""
        return self._home_q.copy()

    def _forward_kinematics(self, q: NDArray[np.float64]) -> SE3Pose:
        """Compute analytical 6-DOF forward kinematics for the articulated manipulator."""
        base_height = 0.22
        l1, l2, l3 = 0.28, 0.26, 0.10

        r01 = _rot_z(float(q[0]))
        r12 = r01 @ _rot_y(float(q[1]))
        p_elbow = np.array([0.0, 0.0, base_height], dtype=np.float64) + r12 @ np.array(
            [l1, 0.0, 0.0], dtype=np.float64
        )

        r23 = r12 @ _rot_y(float(q[2]))
        p_wrist = p_elbow + r23 @ np.array([l2, 0.0, 0.0], dtype=np.float64)

        r34 = r23 @ _rot_x(float(q[3]))
        r45 = r34 @ _rot_y(float(q[4]))
        r_ee = r45 @ _rot_z(float(q[5]))
        p_ee = p_wrist + r_ee @ np.array([l3, 0.0, 0.0], dtype=np.float64)

        quat = _rotation_matrix_to_quaternion_wxyz(r_ee)
        return SE3Pose(
            position=(float(p_ee[0]), float(p_ee[1]), float(p_ee[2])),
            quaternion_wxyz=quat,
        )

    def reset(self, seed: int = 42) -> tuple[JointState, SE3Pose, SE3Pose]:
        """Reset manipulator to home configuration and sample a seeded object target pose."""
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        self._q = self._home_q.copy()
        self._dq = np.zeros(self._dof, dtype=np.float64)
        self._torques = np.zeros(self._dof, dtype=np.float64)
        self._episode_id = seed
        self._step_index = 0
        self._object_grasped = False

        ee_home = self._forward_kinematics(self._q)
        hx, hy, hz = ee_home.position
        dx = float(rng.uniform(-0.06, 0.06))
        dy = float(rng.uniform(-0.08, 0.08))
        dz = float(rng.uniform(-0.10, -0.05))
        self._object_pose = SE3Pose(
            position=(hx + dx, hy + dy, hz + dz),
            quaternion_wxyz=ee_home.quaternion_wxyz,
        )

        joint_state = JointState(
            positions_rad=tuple(float(v) for v in self._q),
            velocities_rad_s=tuple(float(v) for v in self._dq),
            torques_nm=tuple(float(v) for v in self._torques),
        )
        return joint_state, ee_home, self._object_pose

    def get_current_ee_pose(self) -> SE3Pose:
        """Return current end-effector SE(3) pose in world coordinates."""
        return self._forward_kinematics(self._q)

    def get_target_pose(self) -> SE3Pose:
        """Return current privileged object SE(3) pose in world coordinates."""
        return self._object_pose

    def get_jacobian(self) -> NDArray[np.float64]:
        """Compute 6x6 spatial geometric Jacobian at the current joint configuration."""
        eps = 1e-6
        jac = np.zeros((6, self._dof), dtype=np.float64)
        base_pose = self._forward_kinematics(self._q)
        t_base = base_pose.to_homogeneous_matrix()
        p_base = t_base[:3, 3]
        r_base = t_base[:3, :3]

        for idx in range(self._dof):
            q_pert = self._q.copy()
            q_pert[idx] += eps
            pert_pose = self._forward_kinematics(q_pert)
            t_pert = pert_pose.to_homogeneous_matrix()
            dp = (t_pert[:3, 3] - p_base) / eps
            drot = so3_log_map(t_pert[:3, :3] @ r_base.T) / eps
            jac[:3, idx] = dp
            jac[3:, idx] = drot

        return jac

    def step(self, joint_targets_rad: tuple[float, ...]) -> EpisodeStepRecord:
        """Advance damped actuator dynamics by one control step."""
        if len(joint_targets_rad) != self._dof:
            msg = f"Expected {self._dof} joint targets; got {len(joint_targets_rad)}"
            raise ValueError(msg)

        q_des = np.asarray(joint_targets_rad, dtype=np.float64)
        q_err = q_des - self._q
        self._torques = self._kp * q_err - self._kd * self._dq
        q_next = self._q + self._tracking_alpha * q_err
        self._dq = (q_next - self._q) / self._control_dt
        self._q = q_next
        self._step_index += 1

        ee_pose = self._forward_kinematics(self._q)
        ee_pos = np.asarray(ee_pose.position, dtype=np.float64)
        obj_pos = np.asarray(self._object_pose.position, dtype=np.float64)

        if not self._object_grasped and float(np.linalg.norm(ee_pos - obj_pos)) <= 0.015:
            self._object_grasped = True

        if self._object_grasped:
            self._object_pose = ee_pose

        joint_state = JointState(
            positions_rad=tuple(float(v) for v in self._q),
            velocities_rad_s=tuple(float(v) for v in self._dq),
            torques_nm=tuple(float(v) for v in self._torques),
        )
        return EpisodeStepRecord(
            episode_id=self._episode_id,
            step_index=self._step_index,
            timestamp_s=round(self._step_index * self._control_dt, 6),
            joint_state=joint_state,
            end_effector_pose=ee_pose,
            privileged_object_pose=self._object_pose,
            action_joint_targets_rad=tuple(float(v) for v in q_des),
            is_terminal=False,
        )
