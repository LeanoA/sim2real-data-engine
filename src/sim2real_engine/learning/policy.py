"""Neural network policy architectures for behavioral cloning and action chunking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sim2real_engine.core.contracts import JointState
from sim2real_engine.server.bridge import PolicyFn


class ResidualBlock(nn.Module):
    """Feedforward residual block with LayerNorm and SiLU activation."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        return cast(torch.Tensor, self.act2(out + residual))


class BehaviorCloningMLP(nn.Module):
    """Deep residual MLP policy for behavioral cloning and action chunk prediction.

    Maps normalized observation vectors (e.g. 12-D joint state & velocity, or 19-D with EE pose)
    to multi-step action chunks:
        obs in R^{obs_dim} -> actions in R^{horizon x action_dim}
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int = 1,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = max(1, int(horizon))
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.output_proj = nn.Linear(hidden_dim, self.horizon * action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting action chunks.

        Args:
            obs: Tensor of shape `(batch_size, obs_dim)`

        Returns:
            Tensor of shape `(batch_size, horizon, action_dim)`
        """
        x = self.input_proj(obs)
        for block in self.blocks:
            x = block(x)
        out = self.output_proj(x)
        batch_size = obs.shape[0]
        return cast(torch.Tensor, out.view(batch_size, self.horizon, self.action_dim))

    def save_checkpoint(
        self,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Serialize model weights and architecture metadata into a PyTorch checkpoint."""
        dest = Path(file_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "horizon": self.horizon,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "metadata": metadata or {},
        }
        torch.save(payload, dest)

    @classmethod
    def load_checkpoint(
        cls,
        file_path: Path | str,
        device: str = "cpu",
    ) -> tuple[BehaviorCloningMLP, dict[str, Any]]:
        """Load a BehaviorCloningMLP policy from a saved checkpoint file."""
        src = Path(file_path)
        if not src.is_file():
            msg = f"Checkpoint not found at {src}"
            raise FileNotFoundError(msg)

        dev = torch.device(device)
        checkpoint = torch.load(src, map_location=dev, weights_only=False)
        model = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            horizon=checkpoint["horizon"],
            hidden_dim=checkpoint["hidden_dim"],
            num_blocks=checkpoint["num_blocks"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(dev)
        model.eval()
        metadata: dict[str, Any] = checkpoint.get("metadata", {})
        return model, metadata

    def to_policy_fn(
        self,
        stats: dict[str, Any],
        device: str = "cpu",
    ) -> PolicyFn:
        """Wrap the neural network into a callable `PolicyFn` for `PolicyInferenceBridge`.

        Args:
            stats: Normalization stats dictionary (from dataset or checkpoint metadata).
            device: Execution device for PyTorch tensor inference.
        """
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        q_mean = np.array(stats["q_mean"], dtype=np.float32)
        q_std = np.array(stats["q_std"], dtype=np.float32)
        v_mean = np.array(stats["v_mean"], dtype=np.float32)
        v_std = np.array(stats["v_std"], dtype=np.float32)
        act_mean = np.array(stats["act_mean"], dtype=np.float32)
        act_std = np.array(stats["act_std"], dtype=np.float32)

        def _policy(observation: JointState, step: int) -> NDArray[np.float64]:
            q_raw = np.array(observation.positions_rad, dtype=np.float32)
            v_raw = np.array(observation.velocities_rad_s, dtype=np.float32)

            q_norm = (q_raw - q_mean) / np.where(q_std < 1e-6, 1.0, q_std)
            v_norm = (v_raw - v_mean) / np.where(v_std < 1e-6, 1.0, v_std)

            if self.obs_dim > len(q_norm) + len(v_norm):
                # Model expects EE pose or extra features; pad with zeros if not supplied
                pad_len = self.obs_dim - (len(q_norm) + len(v_norm))
                obs_np = np.concatenate([q_norm, v_norm, np.zeros(pad_len, dtype=np.float32)])
            else:
                obs_np = np.concatenate([q_norm, v_norm])

            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(dev)
            with torch.no_grad():
                chunk_pred = self.forward(obs_tensor).squeeze(0).cpu().numpy()

            # Denormalize action chunk: shape (horizon, action_dim)
            chunk_denorm = (chunk_pred * act_std) + act_mean
            return np.asarray(chunk_denorm, dtype=np.float64)

        return _policy
