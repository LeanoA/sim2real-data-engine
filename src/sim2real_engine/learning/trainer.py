"""Supervised Behavioral Cloning policy training loop and telemetry."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader, random_split

from sim2real_engine.learning.dataset import LeRobotParquetDataset
from sim2real_engine.learning.policy import BehaviorCloningMLP


class TrainerConfig(BaseModel):
    """Configuration hyperparameters for Behavioral Cloning policy training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=15, ge=1, description="Number of complete training epochs.")
    batch_size: int = Field(default=64, ge=1, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Initial AdamW learning rate.")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="L2 weight decay penalty.")
    val_ratio: float = Field(
        default=0.1, ge=0.0, lt=0.5, description="Fraction held out for validation."
    )
    device: Literal["auto", "cpu", "mps", "cuda"] = Field(
        default="auto", description="Execution hardware device."
    )
    loss_type: Literal["smooth_l1", "mse"] = Field(
        default="smooth_l1", description="Supervised regression loss criterion."
    )
    seed: int = Field(default=42, ge=0, description="Deterministic random seed.")


class TrainingSummary(BaseModel):
    """Telemetry report produced upon training completion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_loss: float = Field(..., ge=0.0)
    final_train_loss: float = Field(..., ge=0.0)
    final_val_loss: float = Field(..., ge=0.0)
    loss_reduction_pct: float = Field(...)
    total_epochs: int = Field(..., ge=1)
    total_samples: int = Field(..., ge=1)
    training_time_s: float = Field(..., ge=0.0)
    throughput_samples_per_s: float = Field(..., ge=0.0)
    device_used: str


def resolve_device(device_choice: str) -> torch.device:
    """Resolve compute device targeting Apple Silicon MPS, NVIDIA CUDA, or CPU fallback."""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_choice)


def train_policy(
    dataset: LeRobotParquetDataset,
    config: TrainerConfig,
    checkpoint_path: Path | str | None = None,
) -> tuple[BehaviorCloningMLP, TrainingSummary]:
    """Train a BehaviorCloningMLP policy on demonstration dataset transitions."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = resolve_device(config.device)

    # Train / validation split
    total_count = len(dataset)
    val_count = int(total_count * config.val_ratio)
    train_count = total_count - val_count

    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = BehaviorCloningMLP(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        horizon=dataset.horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    criterion: nn.Module = nn.SmoothL1Loss() if config.loss_type == "smooth_l1" else nn.MSELoss()

    initial_loss: float = 0.0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0

    start_time = time.perf_counter()
    total_processed_samples = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()
            preds = model(obs_batch)
            if act_batch.shape[-1] > 6:
                arm_pred = preds[..., :6].contiguous()
                arm_tgt = act_batch[..., :6].contiguous()
                grip_pred = preds[..., 6:].contiguous()
                grip_tgt = act_batch[..., 6:].contiguous()
                arm_loss = criterion(arm_pred, arm_tgt)
                grip_loss = criterion(grip_pred, grip_tgt)
                loss = arm_loss + 5.0 * grip_loss
            else:
                loss = criterion(preds, act_batch)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            total_processed_samples += obs_batch.shape[0]

            if epoch == 0 and len(epoch_losses) == 1:
                initial_loss = loss_val

        scheduler.step()
        final_train_loss = float(np.mean(epoch_losses))

        # Evaluate validation loss
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                preds = model(obs_batch)
                if act_batch.shape[-1] > 6:
                    arm_pred = preds[..., :6].contiguous()
                    arm_tgt = act_batch[..., :6].contiguous()
                    grip_pred = preds[..., 6:].contiguous()
                    grip_tgt = act_batch[..., 6:].contiguous()
                    arm_vloss = criterion(arm_pred, arm_tgt)
                    grip_vloss = criterion(grip_pred, grip_tgt)
                    val_losses.append(float((arm_vloss + 5.0 * grip_vloss).item()))
                else:
                    val_losses.append(float(criterion(preds, act_batch).item()))
        final_val_loss = float(np.mean(val_losses)) if val_losses else final_train_loss

    elapsed = max(1e-4, time.perf_counter() - start_time)
    throughput = total_processed_samples / elapsed
    loss_reduction = ((initial_loss - final_val_loss) / max(1e-6, initial_loss)) * 100.0

    summary = TrainingSummary(
        initial_loss=round(initial_loss, 6),
        final_train_loss=round(final_train_loss, 6),
        final_val_loss=round(final_val_loss, 6),
        loss_reduction_pct=round(loss_reduction, 2),
        total_epochs=config.epochs,
        total_samples=total_count,
        training_time_s=round(elapsed, 4),
        throughput_samples_per_s=round(throughput, 1),
        device_used=str(device),
    )

    if checkpoint_path is not None:
        meta = dataset.get_stats_dict()
        meta["training_summary"] = summary.model_dump()
        model.save_checkpoint(checkpoint_path, metadata=meta)

    return model, summary
