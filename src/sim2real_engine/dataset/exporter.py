"""LeRobotDataset v2 chunked tabular Parquet exporter with SHA-256 envelope validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sim2real_engine.core.contracts import EpisodeStepRecord
from sim2real_engine.dataset.manifest import DatasetManifest, FeatureStats


class LeRobotEpisodeExporter:
    """Exports episode rollouts to LeRobotDataset v2 Parquet tables with SHA-256 verification."""

    def __init__(self, root_dir: Path, dataset_name: str = "sim2real_pick_place") -> None:
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self._episode_sha256: dict[str, str] = {}
        self._all_steps: list[EpisodeStepRecord] = []
        self._episode_ids: set[int] = set()

    def write_episode(self, episode_id: int, steps: Sequence[EpisodeStepRecord]) -> Path:
        """Write a single episode rollout trajectory to a chunked Parquet table."""
        if not steps:
            msg = f"Episode {episode_id} must contain at least one step; got empty sequence."
            raise ValueError(msg)

        data_dir = self.root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        filename = f"episode_{episode_id:06d}.parquet"
        file_path = data_dir / filename
        rel_path = f"data/{filename}"

        table = pa.Table.from_pydict(
            {
                "episode_index": [s.episode_id for s in steps],
                "frame_index": [s.step_index for s in steps],
                "timestamp": [s.timestamp_s for s in steps],
                "observation.state": [list(s.joint_state.positions_rad) for s in steps],
                "observation.velocity": [list(s.joint_state.velocities_rad_s) for s in steps],
                "observation.ee_position": [list(s.end_effector_pose.position) for s in steps],
                "observation.ee_quaternion": [
                    list(s.end_effector_pose.quaternion_wxyz) for s in steps
                ],
                "observation.object_position": [
                    list(s.privileged_object_pose.position) for s in steps
                ],
                "observation.object_quaternion": [
                    list(s.privileged_object_pose.quaternion_wxyz) for s in steps
                ],
                "observation.gripper_state": [
                    [1.0 if s.gripper_action and s.gripper_action > 0.5 else 0.0]
                    for s in steps
                ],
                "action": [list(s.action) for s in steps],
                "is_terminal": [s.is_terminal for s in steps],
            }
        )
        pq.write_table(table, file_path)

        # Cryptographic SHA-256 seal
        file_bytes = file_path.read_bytes()
        digest = hashlib.sha256(file_bytes).hexdigest()
        self._episode_sha256[rel_path] = digest
        self._all_steps.extend(steps)
        self._episode_ids.add(episode_id)

        return file_path

    def _compute_stats(self, values: list[list[float]]) -> FeatureStats:
        arr = np.asarray(values, dtype=np.float64)
        mean_vec = tuple(float(x) for x in np.mean(arr, axis=0))
        std_vec = tuple(float(x) for x in np.std(arr, axis=0))
        min_vec = tuple(float(x) for x in np.min(arr, axis=0))
        max_vec = tuple(float(x) for x in np.max(arr, axis=0))
        return FeatureStats(mean=mean_vec, std=std_vec, min=min_vec, max=max_vec)

    def finalize_manifest(self) -> DatasetManifest:
        """Compute empirical feature normalization statistics and write meta/manifest.json."""
        if not self._all_steps:
            msg = "Cannot finalize manifest: no episode steps have been written."
            raise ValueError(msg)

        stats_dict = {
            "observation.state": self._compute_stats(
                [list(s.joint_state.positions_rad) for s in self._all_steps]
            ),
            "observation.velocity": self._compute_stats(
                [list(s.joint_state.velocities_rad_s) for s in self._all_steps]
            ),
            "observation.object_position": self._compute_stats(
                [list(s.privileged_object_pose.position) for s in self._all_steps]
            ),
            "observation.gripper_state": self._compute_stats(
                [
                    [1.0 if s.gripper_action and s.gripper_action > 0.5 else 0.0]
                    for s in self._all_steps
                ]
            ),
            "action": self._compute_stats([list(s.action) for s in self._all_steps]),
        }

        manifest = DatasetManifest(
            dataset_name=self.dataset_name,
            format_version="lerobot_v2",
            num_episodes=len(self._episode_ids),
            total_steps=len(self._all_steps),
            episode_sha256=dict(sorted(self._episode_sha256.items())),
            feature_stats=stats_dict,
        )

        meta_dir = self.root_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = meta_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")

        return manifest

    def verify_integrity(self) -> bool:
        """Verify SHA-256 cryptographic hashes of all exported episode files."""
        manifest_file = self.root_dir / "meta" / "manifest.json"
        if not manifest_file.is_file():
            return False

        try:
            raw_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifest = DatasetManifest.model_validate(raw_manifest)
        except Exception:
            return False

        if not manifest.episode_sha256:
            return False

        for rel_path, expected_digest in manifest.episode_sha256.items():
            full_path = self.root_dir / rel_path
            if not full_path.is_file():
                return False
            actual_digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                return False

        return True
