"""Tests for physical joint jerk, FFT spectral chatter, energetics, contact, and slip."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim2real_engine.benchmarks import (
    compute_episode_control_diagnostics,
    compute_fft_spectral_metrics,
    compute_physical_joint_jerk,
    evaluate_ood_robustness,
)
from sim2real_engine.core.contracts import DomainRandomizationConfig, SE3Pose
from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.randomization.samplers import DomainRandomizer
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert
from sim2real_engine.simulators.mujoco_native import MuJoCoPhysicsAdapter


def test_fft_spectral_metrics_separates_low_and_high_frequency_chatter() -> None:
    """Action-velocity FFT cleanly separates 1 Hz smooth motion from 12 Hz chatter."""
    dt = 0.02
    t = np.arange(100, dtype=np.float64) * dt

    # Integrate sinusoidal velocities to mimic position command sequences
    low_freq_pos = np.sin(2.0 * math.pi * 1.0 * t)[:, np.newaxis] * np.ones((1, 6))
    high_freq_pos = np.sin(2.0 * math.pi * 12.0 * t)[:, np.newaxis] * np.ones((1, 6))

    hf_low, fc_low = compute_fft_spectral_metrics(low_freq_pos, dt=dt, hf_cutoff_hz=5.0)
    hf_high, fc_high = compute_fft_spectral_metrics(high_freq_pos, dt=dt, hf_cutoff_hz=5.0)

    assert hf_low < 0.05
    assert 0.5 <= fc_low <= 2.0
    assert hf_high > 0.90
    assert 10.5 <= fc_high <= 13.5


def test_compute_physical_joint_jerk_matches_analytical_cubic() -> None:
    """2nd central difference of qvel recovers constant physical joint jerk d^3q/dt^3."""
    dt = 0.02
    t = np.arange(50, dtype=np.float64) * dt
    j0 = np.array([2.0, -1.5, 0.5, 0.0, 1.0, -0.5], dtype=np.float64)
    # v(t) = 0.5 * j0 * t^2 -> d^2 v / dt^2 = j0
    velocities = 0.5 * (t[:, np.newaxis] ** 2) * j0[np.newaxis, :]

    jerk_est = compute_physical_joint_jerk(velocities, dt=dt)
    assert jerk_est == pytest.approx(float(np.linalg.norm(j0)), abs=1e-4)


def test_in_hand_slip_zero_for_rigid_transport_nonzero_for_sliding() -> None:
    """TCP-frame slip metric distinguishes rigid grasped motion from in-hand sliding."""
    n = 20
    dt = 0.02
    # Arm moves 10 mm/step along +Y while rotated 90 deg around Z
    q_ee = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    ee_poses = [
        SE3Pose(position=(0.35, -0.05 + 0.01 * i, 0.15), quaternion_wxyz=q_ee) for i in range(n)
    ]
    rigid_obj_poses = [
        SE3Pose(position=(0.35, -0.05 + 0.01 * i, 0.12), quaternion_wxyz=q_ee) for i in range(n)
    ]
    sliding_obj_poses = [
        SE3Pose(position=(0.35, -0.05 + 0.01 * i, 0.12 - 0.002 * i), quaternion_wxyz=q_ee)
        for i in range(n)
    ]

    actions = np.zeros((n, 6), dtype=np.float64)
    vels = np.zeros((n, 6), dtype=np.float64)
    taus = np.zeros((n, 6), dtype=np.float64)
    grips = [1.0] * n

    diag_rigid = compute_episode_control_diagnostics(
        actions=actions,
        joint_velocities=vels,
        joint_torques=taus,
        ee_poses=ee_poses,
        obj_poses=rigid_obj_poses,
        gripper_states=grips,
        dt=dt,
    )
    diag_slip = compute_episode_control_diagnostics(
        actions=actions,
        joint_velocities=vels,
        joint_torques=taus,
        ee_poses=ee_poses,
        obj_poses=sliding_obj_poses,
        gripper_states=grips,
        dt=dt,
    )

    assert diag_rigid.mean_in_hand_slip_mm_s == pytest.approx(0.0, abs=1e-3)
    # 0.002 m / 0.02 s = 0.1 m/s = 100 mm/s
    assert diag_slip.mean_in_hand_slip_mm_s == pytest.approx(100.0, abs=0.5)


def test_mujoco_contact_force_energy_and_gravity_comp_uncertainty() -> None:
    """MuJoCo adapter emits touch forces, energy, and responds to gravity_comp_gain."""
    sim = MuJoCoPhysicsAdapter()
    expert = PrivilegedPickPlaceExpert(ik_solver=DampedLeastSquaresIK(dof=sim.dof))
    nom_records = expert.run_episode(backend=sim, seed=42, max_steps=120)

    forces = [r.contact_force_n for r in nom_records if r.contact_force_n is not None]
    energies = [r.mechanical_energy_j for r in nom_records if r.mechanical_energy_j is not None]
    assert len(forces) == len(nom_records)
    assert len(energies) == len(nom_records)
    assert max(forces) > 0.1
    assert all(math.isfinite(e) for e in energies)

    # Undercompensated feedforward gravity (gain=0.5) forces PD actuator torque to increase
    under_cfg = DomainRandomizationConfig(seed=42, gravity_comp_gain_range=(0.5, 0.5))
    pert = DomainRandomizer(under_cfg).sample_episode(0)
    assert pert.gravity_comp_gain == pytest.approx(0.5, abs=1e-9)

    sim.reset(seed=42, perturbation=pert)
    rec_under = sim.step(tuple(float(v) for v in sim.home_configuration))

    sim.reset(seed=42, perturbation=None)
    rec_nom = sim.step(tuple(float(v) for v in sim.home_configuration))

    tau_under = np.linalg.norm(rec_under.joint_state.torques_nm)
    tau_nom = np.linalg.norm(rec_nom.joint_state.torques_nm)
    assert tau_under > tau_nom + 0.5


def test_ood_benchmark_populates_control_and_spectral_diagnostics() -> None:
    """OOD benchmark collects physical jerk, FFT spectral metrics, work, contact, and slip."""
    report = evaluate_ood_robustness(
        model_path="checkpoints/bc_pick_place.pt",
        backend="mujoco_native",
        num_episodes=2,
        base_seed=42,
        max_steps=20,
        compare_ensembling=True,
    )
    diag = report.nominal.control_diagnostics
    assert diag is not None
    assert diag.mean_physical_joint_jerk_rad_s3 >= 0.0
    assert 0.0 <= diag.mean_hf_power_ratio <= 1.0
    assert diag.mean_mechanical_work_j >= 0.0
    assert diag.mean_grasp_contact_force_n is not None
    assert report.ood_dr_no_ensemble is not None
    assert report.ood_dr_no_ensemble.control_diagnostics is not None
    assert report.hf_power_reduction_pct is not None
    assert report.physical_jerk_reduction_pct is not None
