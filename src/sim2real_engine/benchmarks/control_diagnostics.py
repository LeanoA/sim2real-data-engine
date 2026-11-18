"""Control-theoretic, spectral (FFT), energetics, contact force, and slip diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.benchmarks.statistics import DistributionStats, summarize_distribution
from sim2real_engine.core.contracts import SE3Pose
from sim2real_engine.learning.dataset import transform_to_tcp_frame


class ControlDiagnosticsSummary(BaseModel):
    """Per-episode physical control, spectral chatter, energetics, and grasp contact telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_command_delta_l2: float = Field(
        ..., ge=0.0, description="Mean 1st-order action step delta ||a_t - a_{t-1}||_2 (rad)."
    )
    mean_physical_joint_jerk_rad_s3: float = Field(
        ..., ge=0.0, description="Mean L2 norm of true physical joint jerk d^3q/dt^3 (rad/s^3)."
    )
    rms_joint_torque_nm: float = Field(
        ..., ge=0.0, description="Root-mean-square joint actuator torque across episode (N*m)."
    )
    mechanical_work_proxy_j: float = Field(
        ..., ge=0.0, description="Integrated mechanical work sum(|tau . q_dot| * dt) (J)."
    )
    action_high_freq_power_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of AC action-velocity FFT power strictly >= 5 Hz.",
    )
    action_spectral_centroid_hz: float = Field(
        ..., ge=0.0, description="Power-weighted mean frequency of action-velocity spectrum (Hz)."
    )
    mean_grasp_contact_force_n: float | None = Field(
        default=None, ge=0.0, description="Mean total touch force during active grasp (N)."
    )
    peak_grasp_contact_force_n: float | None = Field(
        default=None, ge=0.0, description="Peak total fingertip touch force across episode (N)."
    )
    mean_in_hand_slip_mm_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Mean TCP-frame object slip speed while fingers are closed (mm/s).",
    )


class RegimeControlDiagnostics(BaseModel):
    """Aggregated control, spectral, energetics, and contact stats across an evaluation regime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_physical_joint_jerk_rad_s3: float = Field(..., ge=0.0)
    mean_rms_torque_nm: float = Field(..., ge=0.0)
    mean_mechanical_work_j: float = Field(..., ge=0.0)
    mean_hf_power_ratio: float = Field(..., ge=0.0, le=1.0)
    mean_spectral_centroid_hz: float = Field(..., ge=0.0)
    mean_grasp_contact_force_n: float | None = None
    mean_peak_contact_force_n: float | None = None
    mean_in_hand_slip_mm_s: float | None = None
    physical_jerk_stats: DistributionStats | None = None
    hf_power_ratio_stats: DistributionStats | None = None
    mechanical_work_stats: DistributionStats | None = None
    contact_force_stats: DistributionStats | None = None
    in_hand_slip_stats: DistributionStats | None = None


def compute_fft_spectral_metrics(
    actions: np.ndarray,
    *,
    dt: float = 0.02,
    hf_cutoff_hz: float = 5.0,
) -> tuple[float, float]:
    """Compute (high_freq_power_ratio, spectral_centroid_hz) from action-velocity FFT."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] == 0:
        return 0.0, 0.0

    # Differentiate position commands -> velocity avoids endpoint ramp leakage in FFT
    vel = np.diff(arr, axis=0) / dt
    n_samples = vel.shape[0]
    vel = vel - np.mean(vel, axis=0, keepdims=True)
    window = np.hanning(n_samples)[:, np.newaxis]
    vel_win = vel * window

    freqs = np.fft.rfftfreq(n_samples, d=dt)
    spec = np.fft.rfft(vel_win, axis=0)
    psd = np.sum(np.abs(spec) ** 2, axis=1)

    ac_mask = freqs > 1e-9
    total_power = float(np.sum(psd[ac_mask]))
    if total_power <= 1e-12:
        return 0.0, 0.0

    hf_mask = freqs >= hf_cutoff_hz
    hf_ratio = float(np.clip(np.sum(psd[hf_mask]) / total_power, 0.0, 1.0))
    centroid_hz = float(np.sum(freqs[ac_mask] * psd[ac_mask]) / total_power)
    return hf_ratio, centroid_hz


def compute_physical_joint_jerk(velocities: np.ndarray, *, dt: float = 0.02) -> float:
    """Compute mean L2 norm of physical joint jerk d^3q/dt^3 via 2nd central diff of qvel."""
    vel = np.asarray(velocities, dtype=np.float64)
    if vel.ndim != 2 or vel.shape[0] < 3 or vel.shape[1] == 0:
        return 0.0

    jerk = (vel[2:] - 2.0 * vel[1:-1] + vel[:-2]) / (dt * dt)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def compute_episode_control_diagnostics(
    *,
    actions: Sequence[Sequence[float]] | Sequence[np.ndarray] | np.ndarray,
    joint_velocities: Sequence[Sequence[float]] | Sequence[np.ndarray] | np.ndarray,
    joint_torques: Sequence[Sequence[float]] | Sequence[np.ndarray] | np.ndarray,
    ee_poses: Sequence[SE3Pose] | None = None,
    obj_poses: Sequence[SE3Pose] | None = None,
    gripper_states: Sequence[float] | None = None,
    contact_forces: Sequence[float | None] | None = None,
    dt: float = 0.02,
    hf_cutoff_hz: float = 5.0,
) -> ControlDiagnosticsSummary:
    """Compute full control, spectral, energetics, contact force, and slip diagnostics."""
    act_arr = np.asarray(actions, dtype=np.float64)
    vel_arr = np.asarray(joint_velocities, dtype=np.float64)
    tau_arr = np.asarray(joint_torques, dtype=np.float64)

    cmd_delta = (
        float(np.mean(np.linalg.norm(np.diff(act_arr, axis=0), axis=1)))
        if act_arr.ndim == 2 and act_arr.shape[0] >= 2
        else 0.0
    )
    phys_jerk = compute_physical_joint_jerk(vel_arr, dt=dt)
    rms_tau = float(np.sqrt(np.mean(tau_arr * tau_arr))) if tau_arr.size > 0 else 0.0

    n_dyn = (
        min(vel_arr.shape[0], tau_arr.shape[0])
        if vel_arr.ndim == 2 and tau_arr.ndim == 2
        else 0
    )
    work_j = (
        float(np.sum(np.abs(np.sum(tau_arr[:n_dyn] * vel_arr[:n_dyn], axis=1))) * dt)
        if n_dyn > 0
        else 0.0
    )

    hf_ratio, centroid_hz = compute_fft_spectral_metrics(
        act_arr, dt=dt, hf_cutoff_hz=hf_cutoff_hz
    )

    # Fingertip normal contact force (when backend reports touch sensor force)
    mean_force: float | None = None
    peak_force: float | None = None
    if contact_forces is not None:
        valid_forces = [float(f) for f in contact_forces if f is not None]
        if valid_forces:
            peak_force = round(max(valid_forces), 4)
            if gripper_states is not None and len(gripper_states) == len(contact_forces):
                grasp_forces = [
                    float(f)
                    for f, g in zip(contact_forces, gripper_states, strict=True)
                    if f is not None and (g > 0.5 or f > 0.05)
                ]
            else:
                grasp_forces = [f for f in valid_forces if f > 0.05]
            mean_force = round(float(np.mean(grasp_forces)) if grasp_forces else 0.0, 4)

    # In-hand TCP-frame slip speed during closed-gripper phase
    mean_slip_mm_s: float | None = None
    if (
        ee_poses is not None
        and obj_poses is not None
        and gripper_states is not None
        and len(ee_poses) == len(obj_poses) == len(gripper_states)
        and len(ee_poses) >= 2
    ):
        ee_pos = np.asarray([p.position for p in ee_poses], dtype=np.float32)
        ee_quat = np.asarray([p.quaternion_wxyz for p in ee_poses], dtype=np.float32)
        obj_pos = np.asarray([p.position for p in obj_poses], dtype=np.float32)
        rel_tcp = transform_to_tcp_frame(obj_pos, ee_pos, ee_quat).astype(np.float64)

        slip_speeds: list[float] = []
        for t in range(1, len(ee_poses)):
            both_closed = gripper_states[t] > 0.5 and gripper_states[t - 1] > 0.5
            in_proximity = float(np.linalg.norm(rel_tcp[t])) <= 0.06
            if both_closed and in_proximity:
                speed_mm_s = float(np.linalg.norm(rel_tcp[t] - rel_tcp[t - 1]) / dt) * 1000.0
                slip_speeds.append(speed_mm_s)
        if slip_speeds:
            mean_slip_mm_s = round(float(np.mean(slip_speeds)), 3)

    return ControlDiagnosticsSummary(
        mean_command_delta_l2=round(cmd_delta, 4),
        mean_physical_joint_jerk_rad_s3=round(phys_jerk, 3),
        rms_joint_torque_nm=round(rms_tau, 4),
        mechanical_work_proxy_j=round(work_j, 4),
        action_high_freq_power_ratio=round(hf_ratio, 4),
        action_spectral_centroid_hz=round(centroid_hz, 3),
        mean_grasp_contact_force_n=mean_force,
        peak_grasp_contact_force_n=peak_force,
        mean_in_hand_slip_mm_s=mean_slip_mm_s,
    )


def aggregate_control_diagnostics(
    episodes: Sequence[ControlDiagnosticsSummary],
    *,
    seed: int = 0,
) -> RegimeControlDiagnostics | None:
    """Aggregate per-episode control & spectral diagnostics with distribution summaries."""
    if not episodes:
        return None

    jerks = [e.mean_physical_joint_jerk_rad_s3 for e in episodes]
    torques = [e.rms_joint_torque_nm for e in episodes]
    works = [e.mechanical_work_proxy_j for e in episodes]
    hf_ratios = [e.action_high_freq_power_ratio for e in episodes]
    centroids = [e.action_spectral_centroid_hz for e in episodes]

    jerk_stats = summarize_distribution(jerks, seed=seed, decimals=3)
    hf_stats = summarize_distribution(hf_ratios, seed=seed, decimals=4)
    work_stats = summarize_distribution(works, seed=seed, decimals=4)

    mean_forces = [
        e.mean_grasp_contact_force_n for e in episodes if e.mean_grasp_contact_force_n is not None
    ]
    peak_forces = [
        e.peak_grasp_contact_force_n for e in episodes if e.peak_grasp_contact_force_n is not None
    ]
    slips = [e.mean_in_hand_slip_mm_s for e in episodes if e.mean_in_hand_slip_mm_s is not None]

    force_stats = (
        summarize_distribution(mean_forces, seed=seed, decimals=3) if mean_forces else None
    )
    slip_stats = summarize_distribution(slips, seed=seed, decimals=3) if slips else None

    return RegimeControlDiagnostics(
        mean_physical_joint_jerk_rad_s3=jerk_stats.mean,
        mean_rms_torque_nm=round(float(np.mean(torques)), 4),
        mean_mechanical_work_j=work_stats.mean,
        mean_hf_power_ratio=hf_stats.mean,
        mean_spectral_centroid_hz=round(float(np.mean(centroids)), 3),
        mean_grasp_contact_force_n=force_stats.mean if force_stats is not None else None,
        mean_peak_contact_force_n=(
            round(float(np.mean(peak_forces)), 3) if peak_forces else None
        ),
        mean_in_hand_slip_mm_s=slip_stats.mean if slip_stats is not None else None,
        physical_jerk_stats=jerk_stats,
        hf_power_ratio_stats=hf_stats,
        mechanical_work_stats=work_stats,
        contact_force_stats=force_stats,
        in_hand_slip_stats=slip_stats,
    )
