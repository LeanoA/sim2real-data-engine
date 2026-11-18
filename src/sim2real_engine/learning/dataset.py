"""LeRobotDataset v2 Parquet loader with feature normalization and action chunking."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from sim2real_engine.dataset.manifest import DatasetManifest


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert scalar-first unit quaternion (w, x, y, z) to 3x3 SO(3) rotation matrix.

    Supports single quaternion of shape (4,) returning (3, 3), or
    batched quaternions of shape (N, 4) returning (N, 3, 3).
    """
    q = np.asarray(quat, dtype=np.float32)
    single = q.ndim == 1
    if single:
        q = q[np.newaxis, :]

    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    q = q / norm

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    r00 = 1.0 - 2.0 * (y**2 + z**2)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)

    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x**2 + z**2)
    r12 = 2.0 * (y * z - w * x)

    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x**2 + y**2)

    r = np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )

    return r[0] if single else r


def transform_to_tcp_frame(
    obj_pos: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
) -> np.ndarray:
    """Transform Cartesian coordinates from world frame to local TCP (end-effector) frame.

    Computes p_obj_tcp = R_ee.T @ (p_obj - p_ee).
    Supports either single vectors ((3,), (3,), (4,)) -> (3,)
    or batched arrays ((N, 3), (N, 3), (N, 4)) -> (N, 3).
    """
    op = np.asarray(obj_pos, dtype=np.float32)
    ep = np.asarray(ee_pos, dtype=np.float32)
    diff = op - ep
    r_ee = quat_to_rot_matrix(ee_quat)

    if diff.ndim == 1:
        return cast(np.ndarray, r_ee.T @ diff)

    return cast(np.ndarray, np.einsum("nji,nj->ni", r_ee, diff))


def compute_relative_quat_tcp(
    obj_quat: np.ndarray,
    ee_quat: np.ndarray,
) -> np.ndarray:
    """Compute canonicalized relative orientation q_rel = q_ee^{-1} * q_obj in TCP frame.

    Enforces positive hemisphere canonicalization (q ~ -q on S^3/Z_2).
    Supports single (4,) -> (4,) or batched (N, 4) -> (N, 4) scalar-first quaternions.
    """
    oq = np.asarray(obj_quat, dtype=np.float32)
    eq = np.asarray(ee_quat, dtype=np.float32)
    single = oq.ndim == 1
    if single:
        oq = oq[np.newaxis, :]
        eq = eq[np.newaxis, :]

    oq_norm = np.linalg.norm(oq, axis=-1, keepdims=True)
    oq = oq / np.where(oq_norm < 1e-8, 1.0, oq_norm)
    eq_norm = np.linalg.norm(eq, axis=-1, keepdims=True)
    eq = eq / np.where(eq_norm < 1e-8, 1.0, eq_norm)

    # Conjugate of ee_quat: (w, -x, -y, -z)
    w1, x1, y1, z1 = eq[:, 0], -eq[:, 1], -eq[:, 2], -eq[:, 3]
    w2, x2, y2, z2 = oq[:, 0], oq[:, 1], oq[:, 2], oq[:, 3]

    rw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    rx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    ry = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    rz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    rel = np.stack([rw, rx, ry, rz], axis=-1)
    rel_norm = np.linalg.norm(rel, axis=-1, keepdims=True)
    rel = rel / np.where(rel_norm < 1e-8, 1.0, rel_norm)

    # Canonicalize antipodal ambiguity q ~ -q onto positive-w hemisphere
    sign = np.where(
        np.abs(rel[:, 0]) > 1e-7,
        np.where(rel[:, 0] < 0.0, -1.0, 1.0),
        np.where(rel[:, 1] < 0.0, -1.0, 1.0),
    ).astype(np.float32)
    rel = rel * sign[:, np.newaxis]

    return rel[0] if single else rel


def assemble_policy_observation(
    q_norm: np.ndarray,
    v_norm: np.ndarray,
    *,
    include_ee_pose: bool = True,
    include_object_pose: bool = True,
    use_tcp_frame: bool = False,
    ego_centric: bool = False,
    ee_pos: np.ndarray | None = None,
    ee_quat: np.ndarray | None = None,
    obj_pos: np.ndarray | None = None,
    obj_quat: np.ndarray | None = None,
    gripper_state: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble single or batched observation vectors consistently across train and inference."""
    parts: list[np.ndarray] = [q_norm, v_norm]

    if ego_centric:
        if (
            ee_pos is not None
            and ee_quat is not None
            and obj_pos is not None
            and obj_quat is not None
        ):
            rel_pos = transform_to_tcp_frame(obj_pos, ee_pos, ee_quat)
            rel_quat = compute_relative_quat_tcp(obj_quat, ee_quat)
            parts.extend([rel_pos, rel_quat])
        if gripper_state is not None:
            parts.append(gripper_state)
        return np.concatenate(parts, axis=-1)

    if include_ee_pose and ee_pos is not None and ee_quat is not None:
        parts.extend([ee_pos, ee_quat])

    if include_object_pose and obj_pos is not None and obj_quat is not None:
        if include_ee_pose and ee_pos is not None:
            if use_tcp_frame and ee_quat is not None:
                rel_pos = transform_to_tcp_frame(obj_pos, ee_pos, ee_quat)
            else:
                rel_pos = obj_pos - ee_pos
            parts.extend([obj_pos, obj_quat, rel_pos])
        else:
            parts.extend([obj_pos, obj_quat])
        if gripper_state is not None:
            parts.append(gripper_state)

    return np.concatenate(parts, axis=-1)


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
        include_object_pose: bool = True,
        use_tcp_frame: bool = False,
        ego_centric: bool = False,
        device: str = "cpu",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.horizon = max(1, int(horizon))
        self.include_ee_pose = include_ee_pose
        self.include_object_pose = include_object_pose
        self.ego_centric = bool(ego_centric)
        self.use_tcp_frame = bool(use_tcp_frame or ego_centric)
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
        self._episode_ids: list[int] = []
        self.num_episodes: int = 0
        self._load_episodes(parquet_files)

        if not self._samples:
            msg = f"Zero valid transitions loaded from {self.root_dir}"
            raise ValueError(msg)

        self.obs_dim = self._samples[0][0].shape[0]

    def _load_episodes(self, files: list[Path]) -> None:
        ep_idx = 0
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

            ee_pos = (
                np.array(d["observation.ee_position"], dtype=np.float32)
                if "observation.ee_position" in d
                else None
            )
            ee_quat = (
                np.array(d["observation.ee_quaternion"], dtype=np.float32)
                if "observation.ee_quaternion" in d
                else None
            )
            obj_pos = (
                np.array(d["observation.object_position"], dtype=np.float32)
                if "observation.object_position" in d
                else None
            )
            obj_quat = (
                np.array(d["observation.object_quaternion"], dtype=np.float32)
                if "observation.object_quaternion" in d
                else None
            )
            grip_state = (
                np.array(d["observation.gripper_state"], dtype=np.float32)
                if "observation.gripper_state" in d
                else None
            )

            obs_all = assemble_policy_observation(
                q_norm,
                v_norm,
                include_ee_pose=self.include_ee_pose,
                include_object_pose=self.include_object_pose,
                use_tcp_frame=self.use_tcp_frame,
                ego_centric=self.ego_centric,
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                obj_pos=obj_pos,
                obj_quat=obj_quat,
                gripper_state=grip_state,
            )

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
                self._episode_ids.append(ep_idx)
            ep_idx += 1

        self.num_episodes = ep_idx

    @property
    def episode_ids(self) -> tuple[int, ...]:
        """Episode index associated with each transition sample."""
        return tuple(self._episode_ids)

    def get_episode_split_indices(
        self, val_ratio: float, seed: int = 42
    ) -> tuple[list[int], list[int], int, int]:
        """Partition transition indices by whole episodes to prevent temporal data leakage.

        Returns (train_indices, val_indices, train_episodes, val_episodes).
        """
        if val_ratio <= 0.0 or self.num_episodes < 2:
            all_idx = list(range(len(self._samples)))
            return all_idx, [], self.num_episodes, 0

        rng = np.random.default_rng(seed)
        perm_episodes = rng.permutation(self.num_episodes)
        val_ep_count = max(1, min(self.num_episodes - 1, int(round(self.num_episodes * val_ratio))))
        val_ep_set = {int(ep) for ep in perm_episodes[:val_ep_count]}

        train_idx: list[int] = []
        val_idx: list[int] = []
        for idx, ep in enumerate(self._episode_ids):
            if ep in val_ep_set:
                val_idx.append(idx)
            else:
                train_idx.append(idx)

        return train_idx, val_idx, self.num_episodes - val_ep_count, val_ep_count

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
        obj_pos: Sequence[float] | np.ndarray | None = None,
        obj_quat: Sequence[float] | np.ndarray | None = None,
        gripper_state: float | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Normalize an incoming runtime observation for policy inference."""
        q = (np.asarray(positions_rad, dtype=np.float32) - self._q_mean) / self._q_std
        v = (np.asarray(velocities_rad_s, dtype=np.float32) - self._v_mean) / self._v_std
        if self.include_ee_pose and (ee_pos is None or ee_quat is None):
            msg = "ee_pos and ee_quat must be provided when include_ee_pose=True"
            raise ValueError(msg)

        return assemble_policy_observation(
            q,
            v,
            include_ee_pose=self.include_ee_pose,
            include_object_pose=self.include_object_pose,
            use_tcp_frame=self.use_tcp_frame,
            ego_centric=self.ego_centric,
            ee_pos=np.asarray(ee_pos, dtype=np.float32) if ee_pos is not None else None,
            ee_quat=np.asarray(ee_quat, dtype=np.float32) if ee_quat is not None else None,
            obj_pos=np.asarray(obj_pos, dtype=np.float32) if obj_pos is not None else None,
            obj_quat=np.asarray(obj_quat, dtype=np.float32) if obj_quat is not None else None,
            gripper_state=(
                np.atleast_1d(np.asarray(gripper_state, dtype=np.float32))
                if gripper_state is not None
                else None
            ),
        )

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
            "include_object_pose": self.include_object_pose,
            "use_tcp_frame": self.use_tcp_frame,
            "ego_centric": self.ego_centric,
        }
