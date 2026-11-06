"""Bit-exact seeded domain randomization samplers with isolated PCG64 PRNG streams."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.core.contracts import DomainRandomizationConfig, SE3Pose


class EpisodePerturbation(BaseModel):
    """Deterministic physical and sensor perturbation parameter set for an episode rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: int = Field(..., ge=0, description="Unique zero-indexed episode identifier.")
    derived_seed: int = Field(
        ...,
        ge=0,
        description="Deterministic 64-bit integer seed derived for this episode child stream.",
    )
    mass_scale: float = Field(
        ...,
        gt=0.0,
        description="Multiplicative payload and link mass scaling factor.",
    )
    friction_scale: float = Field(
        ...,
        gt=0.0,
        description="Multiplicative Coulomb surface friction scaling factor.",
    )
    damping_scale: float = Field(
        ...,
        gt=0.0,
        description="Multiplicative joint damping scaling factor.",
    )
    lighting_intensity: float = Field(
        ...,
        gt=0.0,
        description="Relative scene illumination multiplier.",
    )
    camera_pose_offset: SE3Pose = Field(
        ...,
        description="Camera extrinsic SE(3) spatial perturbation relative to nominal mount.",
    )


class DomainRandomizer:
    """Bit-exact seeded domain randomizer driving isolated PCG64 PRNG child streams."""

    def __init__(
        self,
        config: DomainRandomizationConfig,
        camera_rotation_jitter_rad: float = 0.01,
    ) -> None:
        self.config = config
        self.camera_rotation_jitter_rad = max(0.0, float(camera_rotation_jitter_rad))

    def sample_episode(self, episode_id: int) -> EpisodePerturbation:
        """Sample deterministic physical and extrinsic perturbations for an episode."""
        if episode_id < 0:
            msg = f"Episode index must be non-negative; got {episode_id}"
            raise ValueError(msg)

        # Derive isolated deterministic child sequence
        seed_seq = np.random.SeedSequence([self.config.seed, episode_id])
        derived_seed = int(seed_seq.generate_state(1, dtype=np.uint64)[0])

        # Backed strictly by PCG64 bit generator
        bit_generator = np.random.PCG64(seed_seq)
        rng = np.random.default_rng(bit_generator)

        mass_min, mass_max = self.config.mass_scale_range
        fric_min, fric_max = self.config.friction_scale_range
        damp_min, damp_max = self.config.damping_scale_range
        light_min, light_max = self.config.lighting_intensity_range

        mass_scale = float(rng.uniform(mass_min, mass_max))
        friction_scale = float(rng.uniform(fric_min, fric_max))
        damping_scale = float(rng.uniform(damp_min, damp_max))
        lighting_intensity = float(rng.uniform(light_min, light_max))

        # Camera position jitter: uniform in [-delta_c, +delta_c]^3
        delta_c = self.config.camera_position_jitter_m
        if delta_c > 0.0:
            pos_offset = (
                float(rng.uniform(-delta_c, delta_c)),
                float(rng.uniform(-delta_c, delta_c)),
                float(rng.uniform(-delta_c, delta_c)),
            )
        else:
            pos_offset = (0.0, 0.0, 0.0)

        # Camera orientation jitter: Lie algebra exponential map from so(3) vector
        if self.camera_rotation_jitter_rad > 0.0:
            delta_theta = rng.normal(0.0, self.camera_rotation_jitter_rad, size=3).astype(
                np.float64
            )
            theta = float(np.linalg.norm(delta_theta))
            if theta < 1e-12:
                quat_wxyz = (1.0, 0.0, 0.0, 0.0)
            else:
                axis = delta_theta / theta
                half_theta = theta * 0.5
                qw = math.cos(half_theta)
                qx = float(axis[0] * math.sin(half_theta))
                qy = float(axis[1] * math.sin(half_theta))
                qz = float(axis[2] * math.sin(half_theta))
                quat_wxyz = (float(qw), qx, qy, qz)
        else:
            quat_wxyz = (1.0, 0.0, 0.0, 0.0)

        camera_pose_offset = SE3Pose.from_unnormalized(
            position=pos_offset,
            quaternion_wxyz=quat_wxyz,
        )

        return EpisodePerturbation(
            episode_id=episode_id,
            derived_seed=derived_seed,
            mass_scale=mass_scale,
            friction_scale=friction_scale,
            damping_scale=damping_scale,
            lighting_intensity=lighting_intensity,
            camera_pose_offset=camera_pose_offset,
        )

    def sample_batch(self, num_episodes: int) -> list[EpisodePerturbation]:
        """Sample a deterministic batch of perturbations for episodes 0..num_episodes-1."""
        if num_episodes < 0:
            msg = f"Number of episodes must be non-negative; got {num_episodes}"
            raise ValueError(msg)
        return [self.sample_episode(i) for i in range(num_episodes)]
