"""SO(3)/SE(3) spatial math, quaternion SLERP, and Damped Least-Squares Differential IK."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from sim2real_engine.core.contracts import SE3Pose


def so3_log_map(rotation_error: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the logarithmic map on SO(3), mapping a 3x3 rotation matrix to so(3) in R^3."""
    if rotation_error.shape != (3, 3):
        msg = f"Expected (3, 3) rotation matrix; got {rotation_error.shape}"
        raise ValueError(msg)

    trace_val = float(np.trace(rotation_error))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace_val - 1.0)))
    theta = math.acos(cos_theta)

    vee = np.array(
        [
            rotation_error[2, 1] - rotation_error[1, 2],
            rotation_error[0, 2] - rotation_error[2, 0],
            rotation_error[1, 0] - rotation_error[0, 1],
        ],
        dtype=np.float64,
    )

    if theta < 1e-7:
        return 0.5 * vee

    if math.pi - theta < 1e-5:
        diag = np.maximum(0.0, 0.5 * (np.diag(rotation_error) + 1.0))
        axis = np.sqrt(diag)
        if vee[0] < 0.0:
            axis[0] = -axis[0]
        if vee[1] < 0.0:
            axis[1] = -axis[1]
        if vee[2] < 0.0:
            axis[2] = -axis[2]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-12:
            axis = axis / axis_norm
        return theta * axis

    scale = theta / (2.0 * math.sin(theta))
    return scale * vee


def quat_slerp(
    q0_wxyz: tuple[float, float, float, float],
    q1_wxyz: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Spherical linear interpolation on S^3 with antipodal dot-product correction."""
    if not 0.0 <= alpha <= 1.0:
        msg = f"Interpolation parameter alpha must be in [0.0, 1.0]; got {alpha}"
        raise ValueError(msg)

    q0 = np.asarray(q0_wxyz, dtype=np.float64)
    q1 = np.asarray(q1_wxyz, dtype=np.float64)
    q0 = q0 / float(np.linalg.norm(q0))
    q1 = q1 / float(np.linalg.norm(q1))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        interp = (1.0 - alpha) * q0 + alpha * q1
        interp = interp / float(np.linalg.norm(interp))
    else:
        theta_0 = math.acos(min(1.0, dot))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * alpha
        s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        interp = s0 * q0 + s1 * q1
        interp = interp / float(np.linalg.norm(interp))

    return (float(interp[0]), float(interp[1]), float(interp[2]), float(interp[3]))


def compute_se3_error(current: SE3Pose, target: SE3Pose) -> NDArray[np.float64]:
    """Compute 6D spatial twist error [dp_x, dp_y, dp_z, dw_x, dw_y, dw_z] in world frame."""
    t_cur = current.to_homogeneous_matrix()
    t_des = target.to_homogeneous_matrix()

    pos_error = t_des[:3, 3] - t_cur[:3, 3]
    rot_error_matrix = t_des[:3, :3] @ t_cur[:3, :3].T
    rot_error = so3_log_map(rot_error_matrix)

    twist_error = np.zeros(6, dtype=np.float64)
    twist_error[:3] = pos_error
    twist_error[3:] = rot_error
    return twist_error


class DampedLeastSquaresIK:
    """Singularity-robust Levenberg-Marquardt differential IK with null-space posture control."""

    def __init__(
        self,
        dof: int,
        lambda_min: float = 1e-3,
        lambda_max: float = 1e-1,
        sigma_thresh: float = 5e-2,
        k_null: float = 0.5,
    ) -> None:
        if dof < 1:
            msg = f"DOF must be >= 1; got {dof}"
            raise ValueError(msg)
        if lambda_min <= 0.0 or lambda_max < lambda_min:
            msg = f"Invalid damping bounds: lambda_min={lambda_min}, lambda_max={lambda_max}"
            raise ValueError(msg)
        self.dof = dof
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.sigma_thresh = sigma_thresh
        self.k_null = k_null

    def compute_adaptive_damping(self, jacobian: NDArray[np.float64]) -> float:
        """Compute adaptive damping factor lambda based on minimum singular value of J."""
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        sigma_min = float(np.min(singular_values))
        if sigma_min < self.sigma_thresh:
            ratio = sigma_min / self.sigma_thresh
            damping = self.lambda_max * math.sqrt(max(0.0, 1.0 - ratio * ratio))
            return max(self.lambda_min, damping)
        return self.lambda_min

    def step(
        self,
        q_current: NDArray[np.float64],
        q_home: NDArray[np.float64],
        current_pose: SE3Pose,
        target_pose: SE3Pose,
        jacobian: NDArray[np.float64],
        max_step_rad: float = 0.1,
    ) -> NDArray[np.float64]:
        """Compute bounded joint configuration increment delta_q for a single control step."""
        if q_current.shape != (self.dof,) or q_home.shape != (self.dof,):
            msg = f"Expected q vectors of shape ({self.dof},); got {q_current.shape}"
            raise ValueError(msg)
        if jacobian.shape != (6, self.dof):
            msg = f"Expected Jacobian of shape (6, {self.dof}); got {jacobian.shape}"
            raise ValueError(msg)

        error_6d = compute_se3_error(current_pose, target_pose)
        lam = self.compute_adaptive_damping(jacobian)

        jjt = jacobian @ jacobian.T + (lam * lam) * np.eye(6, dtype=np.float64)
        j_pinv_damped = jacobian.T @ np.linalg.solve(jjt, np.eye(6, dtype=np.float64))

        null_projector = np.eye(self.dof, dtype=np.float64) - (j_pinv_damped @ jacobian)
        null_term = null_projector @ (self.k_null * (q_home - q_current))

        delta_q: NDArray[np.float64] = j_pinv_damped @ error_6d + null_term

        max_abs = float(np.max(np.abs(delta_q)))
        if max_abs > max_step_rad and max_abs > 1e-12:
            delta_q = delta_q * (max_step_rad / max_abs)

        return delta_q
