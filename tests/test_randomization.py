"""Verification suite for Phase 3: Seeded Domain Randomization engine with PCG64."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import DomainRandomizationConfig
from sim2real_engine.randomization.samplers import DomainRandomizer, EpisodePerturbation


def test_domain_randomizer_reproducibility() -> None:
    """Independent DomainRandomizer instances with the same seed must produce identical outputs."""
    config1 = DomainRandomizationConfig(seed=42)
    config2 = DomainRandomizationConfig(seed=42)

    dr1 = DomainRandomizer(config=config1)
    dr2 = DomainRandomizer(config=config2)

    batch1 = dr1.sample_batch(10)
    batch2 = dr2.sample_batch(10)

    assert len(batch1) == len(batch2) == 10
    for p1, p2 in zip(batch1, batch2, strict=True):
        assert p1.episode_id == p2.episode_id
        assert p1.derived_seed == p2.derived_seed
        assert math.isclose(p1.mass_scale, p2.mass_scale, abs_tol=1e-12)
        assert math.isclose(p1.friction_scale, p2.friction_scale, abs_tol=1e-12)
        assert math.isclose(p1.damping_scale, p2.damping_scale, abs_tol=1e-12)
        assert math.isclose(p1.lighting_intensity, p2.lighting_intensity, abs_tol=1e-12)
        np.testing.assert_allclose(
            p1.camera_pose_offset.position,
            p2.camera_pose_offset.position,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            p1.camera_pose_offset.quaternion_wxyz,
            p2.camera_pose_offset.quaternion_wxyz,
            atol=1e-12,
        )


def test_out_of_order_stream_isolation() -> None:
    """Episodes sampled out-of-order must match sequential batch results bit-for-bit (ADR-03)."""
    config = DomainRandomizationConfig(seed=1337)
    dr_seq = DomainRandomizer(config=config)
    dr_rand = DomainRandomizer(config=config)

    seq_batch = dr_seq.sample_batch(25)

    # Query in arbitrary non-sequential order
    for ep_id in [19, 4, 24, 0, 11, 7]:
        single_sample = dr_rand.sample_episode(ep_id)
        assert single_sample == seq_batch[ep_id]


def test_statistical_distribution_bounds() -> None:
    """Sampled perturbations over 500 episodes must strictly adhere to configured bounds."""
    config = DomainRandomizationConfig(
        seed=100,
        mass_scale_range=(0.85, 1.15),
        friction_scale_range=(0.70, 1.30),
        damping_scale_range=(0.90, 1.10),
        lighting_intensity_range=(0.60, 1.40),
        camera_position_jitter_m=0.015,
    )
    dr = DomainRandomizer(config=config, camera_rotation_jitter_rad=0.02)
    samples = dr.sample_batch(500)

    masses: list[float] = []
    frictions: list[float] = []
    dampings: list[float] = []
    lightings: list[float] = []

    for p in samples:
        assert 0.85 <= p.mass_scale <= 1.15
        assert 0.70 <= p.friction_scale <= 1.30
        assert 0.90 <= p.damping_scale <= 1.10
        assert 0.60 <= p.lighting_intensity <= 1.40

        x, y, z = p.camera_pose_offset.position
        assert abs(x) <= 0.015 + 1e-9
        assert abs(y) <= 0.015 + 1e-9
        assert abs(z) <= 0.015 + 1e-9

        qw, qx, qy, qz = p.camera_pose_offset.quaternion_wxyz
        quat_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        assert math.isclose(quat_norm, 1.0, abs_tol=1e-5)

        masses.append(p.mass_scale)
        frictions.append(p.friction_scale)
        dampings.append(p.damping_scale)
        lightings.append(p.lighting_intensity)

    # Law of large numbers: empirical means must be close to theoretical midpoints
    assert 0.98 <= float(np.mean(masses)) <= 1.02
    assert 0.98 <= float(np.mean(frictions)) <= 1.02
    assert 0.98 <= float(np.mean(dampings)) <= 1.02
    assert 0.98 <= float(np.mean(lightings)) <= 1.02


def test_zero_camera_jitter_edge_case() -> None:
    """Zero camera jitter produces zero translation and identity quaternion."""
    config = DomainRandomizationConfig(seed=42, camera_position_jitter_m=0.0)
    dr = DomainRandomizer(config=config, camera_rotation_jitter_rad=0.0)
    perturbation = dr.sample_episode(0)

    assert perturbation.camera_pose_offset.position == (0.0, 0.0, 0.0)
    assert perturbation.camera_pose_offset.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_invalid_episode_index_raises() -> None:
    """Negative episode indices or batch sizes must raise ValueError."""
    dr = DomainRandomizer(config=DomainRandomizationConfig(seed=42))
    with pytest.raises(ValueError, match="Episode index must be non-negative"):
        dr.sample_episode(-1)

    with pytest.raises(ValueError, match="Number of episodes must be non-negative"):
        dr.sample_batch(-5)


def test_cli_sample_dr_json(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real sample-dr --seed 42 --episodes 3 --json` outputs valid serialized perturbations."""
    ret = main(["sample-dr", "--seed", "42", "--episodes", "3", "--json"])
    assert ret == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    assert len(payload) == 3

    p0 = EpisodePerturbation.model_validate(payload[0])
    assert p0.episode_id == 0
    assert p0.mass_scale > 0.0


def test_cli_sample_dr_plain_text(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real sample-dr` without --json prints formatted episode summaries."""
    ret = main(["sample-dr", "--seed", "42", "--episodes", "2"])
    assert ret == 0

    captured = capsys.readouterr()
    assert "Sampled 2 perturbations" in captured.out
    assert "- Ep 0:" in captured.out
    assert "- Ep 1:" in captured.out


def test_cli_sample_dr_invalid_episodes(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real sample-dr --episodes 0` exits with non-zero error."""
    ret = main(["sample-dr", "--episodes", "0"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Number of episodes must be >= 1" in captured.err
