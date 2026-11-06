"""Dataset manifest and feature normalization contracts for LeRobotDataset v2."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeatureStats(BaseModel):
    """Per-dimension empirical normalization statistics for a continuous tensor feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: tuple[float, ...] = Field(..., min_length=1, description="Empirical mean vector.")
    std: tuple[float, ...] = Field(
        ...,
        min_length=1,
        description="Empirical standard deviation vector.",
    )
    min: tuple[float, ...] = Field(..., min_length=1, description="Empirical minimum value vector.")
    max: tuple[float, ...] = Field(..., min_length=1, description="Empirical maximum value vector.")

    @model_validator(mode="after")
    def _validate_dimensions_and_ranges(self) -> Self:
        d = len(self.mean)
        if len(self.std) != d or len(self.min) != d or len(self.max) != d:
            msg = (
                f"FeatureStats vectors must share identical dimension; got "
                f"mean={d}, std={len(self.std)}, min={len(self.min)}, max={len(self.max)}"
            )
            raise ValueError(msg)
        for i in range(d):
            if self.std[i] < 0.0:
                msg = f"Standard deviation must be non-negative at index {i}; got {self.std[i]}"
                raise ValueError(msg)
            if self.min[i] > self.max[i]:
                msg = (
                    f"Minimum {self.min[i]} exceeds maximum {self.max[i]} "
                    f"at dimension index {i}"
                )
                raise ValueError(msg)
        return self


class DatasetManifest(BaseModel):
    """Cryptographically sealed LeRobotDataset v2 dataset manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str = Field(..., min_length=1)
    format_version: Literal["lerobot_v2"] = "lerobot_v2"
    num_episodes: int = Field(..., ge=0)
    total_steps: int = Field(..., ge=0)
    episode_sha256: dict[str, str] = Field(
        default_factory=dict,
        description="Relative file path to SHA-256 hex digest mapping.",
    )
    feature_stats: dict[str, FeatureStats] = Field(
        default_factory=dict,
        description="Feature name to empirical normalization stats mapping.",
    )
