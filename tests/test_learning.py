"""Unit and integration tests for Behavioral Cloning policy learning and inference bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import JointState
from sim2real_engine.learning import (
    BehaviorCloningMLP,
    LeRobotParquetDataset,
    TrainerConfig,
    train_policy,
)
from sim2real_engine.server.bridge import PolicyInferenceBridge
from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer


@pytest.fixture
def dataset_path() -> Path:
    p = Path("datasets/sim2real_pick_place")
    if not p.is_dir() or not (p / "meta" / "manifest.json").is_file():
        pytest.skip("Dataset 'datasets/sim2real_pick_place' not found; run export-dataset first.")
    return p


def test_dataset_loading_and_chunking(dataset_path: Path) -> None:
    ds = LeRobotParquetDataset(dataset_path, horizon=4, include_ee_pose=False)
    assert len(ds) > 0
    assert ds.obs_dim == 12
    assert ds.action_dim == 6
    assert ds.horizon == 4

    obs, act = ds[0]
    assert obs.shape == torch.Size([12])
    assert act.shape == torch.Size([4, 6])

    denorm = ds.denormalize_action(act)
    assert denorm.shape == (4, 6)

    norm_obs = ds.normalize_observation(
        positions_rad=tuple(np.zeros(6)),
        velocities_rad_s=tuple(np.zeros(6)),
    )
    assert norm_obs.shape == (12,)


def test_dataset_with_ee_pose(dataset_path: Path) -> None:
    ds = LeRobotParquetDataset(dataset_path, horizon=1, include_ee_pose=True)
    assert ds.obs_dim == 19
    obs, act = ds[0]
    assert obs.shape == torch.Size([19])
    assert act.shape == torch.Size([1, 6])


def test_policy_forward_shape() -> None:
    model = BehaviorCloningMLP(obs_dim=12, action_dim=6, horizon=8, hidden_dim=64, num_blocks=1)
    x = torch.randn(4, 12)
    out = model(x)
    assert out.shape == torch.Size([4, 8, 6])


def test_policy_checkpoint_serialization(tmp_path: Path) -> None:
    model = BehaviorCloningMLP(obs_dim=12, action_dim=6, horizon=4, hidden_dim=64, num_blocks=1)
    ckpt_file = tmp_path / "model.pt"
    meta = {"test_key": "test_val", "q_mean": [0.0] * 6, "act_mean": [0.0] * 6}
    model.save_checkpoint(ckpt_file, metadata=meta)
    assert ckpt_file.is_file()

    loaded, loaded_meta = BehaviorCloningMLP.load_checkpoint(ckpt_file, device="cpu")
    assert loaded.obs_dim == 12
    assert loaded.action_dim == 6
    assert loaded.horizon == 4
    assert loaded_meta["test_key"] == "test_val"


def test_policy_to_policy_fn_bridge_integration(dataset_path: Path) -> None:
    ds = LeRobotParquetDataset(dataset_path, horizon=4, include_ee_pose=False)
    model = BehaviorCloningMLP(obs_dim=ds.obs_dim, action_dim=ds.action_dim, horizon=4)
    stats = ds.get_stats_dict()
    policy_fn = model.to_policy_fn(stats, device="cpu")

    js = JointState(
        positions_rad=(0.0, -0.6, 1.8, 0.0, -0.6, 0.0),
        velocities_rad_s=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        torques_nm=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    predicted_chunk = policy_fn(js, step=0)
    assert predicted_chunk.shape == (4, 6)

    buffer = TemporalEnsembleBuffer(action_dim=6, horizon=4)
    bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)
    action = bridge.infer_step(js, step=0)
    assert action.shape == (6,)


def test_train_policy_loss_descent(dataset_path: Path, tmp_path: Path) -> None:
    ds = LeRobotParquetDataset(dataset_path, horizon=4, include_ee_pose=False)
    cfg = TrainerConfig(
        epochs=3,
        batch_size=128,
        learning_rate=2e-3,
        device="cpu",
        seed=123,
    )
    ckpt_path = tmp_path / "trained_policy.pt"
    model, summary = train_policy(ds, cfg, checkpoint_path=ckpt_path)

    assert summary.total_epochs == 3
    assert summary.final_val_loss < summary.initial_loss
    assert summary.loss_reduction_pct > 0.0
    assert summary.throughput_samples_per_s > 0.0
    assert ckpt_path.is_file()


def test_cli_train_and_eval_policy(
    dataset_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt_path = tmp_path / "cli_model.pt"

    # Train CLI
    train_rc = main(
        [
            "train-policy",
            "--dataset-dir",
            str(dataset_path),
            "--epochs",
            "2",
            "--batch-size",
            "128",
            "--horizon",
            "4",
            "--output-model",
            str(ckpt_path),
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert train_rc == 0
    captured_train = capsys.readouterr().out
    train_payload = json.loads(captured_train)
    assert train_payload["status"] == "success"
    assert ckpt_path.is_file()

    # Eval CLI
    eval_rc = main(
        [
            "eval-policy",
            "--model-path",
            str(ckpt_path),
            "--backend",
            "mujoco",
            "--episodes",
            "2",
            "--max-steps",
            "30",
            "--json",
        ]
    )
    assert eval_rc == 0
    captured_eval = capsys.readouterr().out
    eval_payload = json.loads(captured_eval)
    assert eval_payload["num_episodes"] == 2
    assert eval_payload["backend"] == "mujoco"
