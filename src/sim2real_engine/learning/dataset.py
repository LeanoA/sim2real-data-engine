"""LeRobotDataset v2 Parquet loader with feature normalization and action chunking."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from sim2real_engine.dataset.manifest import DatasetManifest


class LeRobotParquetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset loading LeRobotDataset v2 Parquet demonstration chunks.

    Supports:
    - Low-dimensional state observations (joint positions, velocities, EE Cartesian pose).
    - Z-score feature normalization using empirical stats from `meta/manifest.json`.
    - Receding-horizon action chunking (predicting actions t..t+H-1).
    """

    def __init__(
        self,
        root_dir: Path | str,
        horizon: int = 1,
        include_ee_pose: bool = True,
        device: str = "cpu",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.horizon = max(1, int(horizon))
        self.include_ee_pose = include_ee_pose
        self.device = torch.device(device)

        manifest_path = self.root_dir / "meta" / "manifest.json"
        if not manifest_path.is_file():
            msg = f"Dataset manifest not found at {manifest_path}"
            raise FileNotFoundError(msg)

        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = DatasetManifest.model_validate(raw_manifest)

        # Cache normalization parameters
        q_stats = self.manifest.feature_stats["observation.state"]
        self._q_mean = np.array(q_stats.mean, dtype=np.float32)
        self._q_std = np.array(q_stats.std, dtype=np.float32)
        self._q_std = np.where(self._q_std < 1e-6, 1.0, self._q_std)

        v_stats = self.manifest.feature_stats["observation.velocity"]
        self._v_mean = np.array(v_stats.mean, dtype=np.float32)
        self._v_std = np.array(v_stats.std, dtype=np.float32)
        self._v_std = np.where(self._v_std < 1e-6, 1.0, self._v_std)

        self._act_mean = np.array(self.manifest.feature_stats["action"].mean, dtype=np.float32)
        self._act_std = np.array(self.manifest.feature_stats["action"].std, dtype=np.float32)
        self._act_std = np.where(self._act_std < 1e-6, 1.0, self._act_std)

        self.action_dim = len(self._act_mean)

        # Load all episode Parquet files
        data_dir = self.root_dir / "data"
        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        if not parquet_files:
            msg = f"No Parquet episode files found in {data_dir}"
            raise FileNotFoundError(msg)

        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._load_episodes(parquet_files)

        if not self._samples:
            msg = f"Zero valid transitions loaded from {self.root_dir}"
            raise ValueError(msg)

        self.obs_dim = self._samples[0][0].shape[0]

    def _load_episodes(self, files: list[Path]) -> None:
        for file_path in files:
            table = pq.read_table(file_path)
            d = table.to_pydict()
            n_steps = len(d["frame_index"])
            if n_steps == 0:
                continue

            q_raw = np.array(d["observation.state"], dtype=np.float32)
            v_raw = np.array(d["observation.velocity"], dtype=np.float32)
            act_raw = np.array(d["action"], dtype=np.float32)

            # Normalize features
            q_norm = (q_raw - self._q_mean) / self._q_std
            v_norm = (v_raw - self._v_mean) / self._v_std
            act_norm = (act_raw - self._act_mean) / self._act_std

            if self.include_ee_pose:
                ee_pos = np.array(d["observation.ee_position"], dtype=np.float32)
                ee_quat = np.array(d["observation.ee_quaternion"], dtype=np.float32)
                obs_all = np.concatenate([q_norm, v_norm, ee_pos, ee_quat], axis=-1)
            else:
                obs_all = np.concatenate([q_norm, v_norm], axis=-1)

            # Build action chunks: shape (H, action_dim)
            for t in range(n_steps):
                obs_t = obs_all[t]
                # Chunk from t to t + H
                end_idx = min(t + self.horizon, n_steps)
                chunk = act_norm[t:end_idx]
                if chunk.shape[0] < self.horizon:
                    # Pad repeating final action
                    pad_len = self.horizon - chunk.shape[0]
                    last_act = chunk[-1:]
                    pad = np.repeat(last_act, pad_len, axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)

                self._samples.append((obs_t, chunk))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        obs_np, act_np = self._samples[idx]
        return torch.from_numpy(obs_np), torch.from_numpy(act_np)

    def denormalize_action(self, norm_action: np.ndarray | torch.Tensor) -> np.ndarray:
        """Invert z-score normalization on an action or action chunk."""
        if isinstance(norm_action, torch.Tensor):
            arr = norm_action.detach().cpu().numpy()
        else:
            arr = np.asarray(norm_action, dtype=np.float32)
        return (arr * self._act_std) + self._act_mean

    def normalize_observation(
        self,
        positions_rad: Sequence[float] | np.ndarray,
        velocities_rad_s: Sequence[float] | np.ndarray,
        ee_pos: Sequence[float] | np.ndarray | None = None,
        ee_quat: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize an incoming runtime observation for policy inference."""
        q = (np.asarray(positions_rad, dtype=np.float32) - self._q_mean) / self._q_std
        v = (np.asarray(velocities_rad_s, dtype=np.float32) - self._v_mean) / self._v_std
        if self.include_ee_pose:
            if ee_pos is None or ee_quat is None:
                msg = "ee_pos and ee_quat must be provided when include_ee_pose=True"
                raise ValueError(msg)
            p = np.asarray(ee_pos, dtype=np.float32)
            q_rot = np.asarray(ee_quat, dtype=np.float32)
            return np.concatenate([q, v, p, q_rot], axis=-1)
        return np.concatenate([q, v], axis=-1)

    def get_stats_dict(self) -> dict[str, Any]:
        """Return serializable normalization statistics."""
        return {
            "q_mean": self._q_mean.tolist(),
            "q_std": self._q_std.tolist(),
            "v_mean": self._v_mean.tolist(),
            "v_std": self._v_std.tolist(),
            "act_mean": self._act_mean.tolist(),
            "act_std": self._act_std.tolist(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "horizon": self.horizon,
            "include_ee_pose": self.include_ee_pose,
        }
