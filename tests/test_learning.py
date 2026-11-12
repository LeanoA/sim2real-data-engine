"""Unit and integration tests for Behavioral Cloning policy learning and inference bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import JointState, SE3Pose
from sim2real_engine.learning import (
    BehaviorCloningMLP,
    LeRobotParquetDataset,
    TrainerConfig,
    quat_to_rot_matrix,
    train_policy,
    transform_to_tcp_frame,
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
    ds = LeRobotParquetDataset(
        dataset_path, horizon=4, include_ee_pose=False, include_object_pose=False
    )
    assert len(ds) > 0
    assert ds.obs_dim == 12
    assert ds.action_dim == 7
    assert ds.horizon == 4

    obs, act = ds[0]
    assert obs.shape == torch.Size([12])
    assert act.shape == torch.Size([4, 7])

    denorm = ds.denormalize_action(act)
    assert denorm.shape == (4, 7)

    norm_obs = ds.normalize_observation(
        positions_rad=tuple(np.zeros(6)),
        velocities_rad_s=tuple(np.zeros(6)),
    )
    assert norm_obs.shape == (12,)


def test_dataset_with_ee_pose(dataset_path: Path) -> None:
    ds = LeRobotParquetDataset(
        dataset_path, horizon=1, include_ee_pose=True, include_object_pose=False
    )
    assert ds.obs_dim == 19
    obs, act = ds[0]
    assert obs.shape == torch.Size([19])
    assert act.shape == torch.Size([1, 7])


def test_dataset_vision_conditioned_with_object_pose(dataset_path: Path) -> None:
    ds = LeRobotParquetDataset(
        dataset_path, horizon=1, include_ee_pose=True, include_object_pose=True
    )
    # 6 q + 6 v + 3 ee_pos + 4 ee_quat + 3 obj_pos + 4 obj_quat + 3 rel_pos + 1 gripper = 30
    assert ds.obs_dim == 30
    obs, act = ds[0]
    assert obs.shape == torch.Size([30])
    assert act.shape == torch.Size([1, 7])


def test_policy_forward_shape() -> None:
    model = BehaviorCloningMLP(obs_dim=27, action_dim=7, horizon=8, hidden_dim=64, num_blocks=1)
    x = torch.randn(4, 27)
    out = model(x)
    assert out.shape == torch.Size([4, 8, 7])


def test_policy_checkpoint_serialization(tmp_path: Path) -> None:
    model = BehaviorCloningMLP(obs_dim=27, action_dim=7, horizon=4, hidden_dim=64, num_blocks=1)
    ckpt_file = tmp_path / "model.pt"
    meta = {"test_key": "test_val", "q_mean": [0.0] * 6, "act_mean": [0.0] * 7}
    model.save_checkpoint(ckpt_file, metadata=meta)
    assert ckpt_file.is_file()

    loaded, loaded_meta = BehaviorCloningMLP.load_checkpoint(ckpt_file, device="cpu")
    assert loaded.obs_dim == 27
    assert loaded.action_dim == 7
    assert loaded.horizon == 4
    assert loaded_meta["test_key"] == "test_val"


def test_policy_to_policy_fn_bridge_integration(dataset_path: Path) -> None:
    ds = LeRobotParquetDataset(
        dataset_path, horizon=4, include_ee_pose=False, include_object_pose=False
    )
    model = BehaviorCloningMLP(obs_dim=ds.obs_dim, action_dim=ds.action_dim, horizon=4)
    stats = ds.get_stats_dict()
    policy_fn = model.to_policy_fn(stats, device="cpu")

    js = JointState(
        positions_rad=(0.0, -0.6, 1.8, 0.0, -0.6, 0.0),
        velocities_rad_s=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        torques_nm=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    predicted_chunk = policy_fn(js, 0)
    assert predicted_chunk.shape == (4, 7)

    buffer = TemporalEnsembleBuffer(action_dim=7, horizon=4)
    bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)
    action = bridge.infer_step(js, step=0)
    assert action.shape == (7,)


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


def test_quat_to_rot_matrix_properties() -> None:
    """Verify SO(3) matrix properties: identity, 90-deg yaw, orthogonality, and batching."""
    # Identity quaternion
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r_id = quat_to_rot_matrix(q_id)
    assert r_id.shape == (3, 3)
    assert np.allclose(r_id, np.eye(3), atol=1e-6)

    # +90 deg yaw around Z: w = cos(pi/4), z = sin(pi/4)
    w = float(np.cos(np.pi / 4.0))
    z = float(np.sin(np.pi / 4.0))
    q_yaw = np.array([w, 0.0, 0.0, z], dtype=np.float32)
    r_yaw = quat_to_rot_matrix(q_yaw)
    expected_r = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(r_yaw, expected_r, atol=1e-6)

    # Orthogonality: R @ R.T == I and det(R) == 1
    assert np.allclose(r_yaw @ r_yaw.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(r_yaw), 1.0, atol=1e-6)

    # Batch support
    q_batch = np.stack([q_id, q_yaw])
    r_batch = quat_to_rot_matrix(q_batch)
    assert r_batch.shape == (2, 3, 3)
    assert np.allclose(r_batch[0], r_id, atol=1e-6)
    assert np.allclose(r_batch[1], r_yaw, atol=1e-6)


def test_transform_to_tcp_frame_spatial_equivariance() -> None:
    """Verify TCP-frame relative transformation under rotation and translation."""
    # End effector at (0.2, 0.3, 0.4) rotated +90 deg yaw around Z
    w = float(np.cos(np.pi / 4.0))
    z = float(np.sin(np.pi / 4.0))
    ee_quat = np.array([w, 0.0, 0.0, z], dtype=np.float32)
    ee_pos = np.array([0.2, 0.3, 0.4], dtype=np.float32)

    # Object is 1 meter ahead of EE along world +Y -> in EE body frame, it must be along +X!
    obj_pos = np.array([0.2, 1.3, 0.4], dtype=np.float32)
    tcp_rel = transform_to_tcp_frame(obj_pos, ee_pos, ee_quat)
    assert tcp_rel.shape == (3,)
    assert np.allclose(tcp_rel, [1.0, 0.0, 0.0], atol=1e-6)

    # Batched transform
    ee_pos_b = np.stack([ee_pos, np.zeros(3, dtype=np.float32)])
    ee_quat_b = np.stack([ee_quat, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)])
    obj_pos_b = np.stack([obj_pos, np.array([0.0, 0.0, 0.5], dtype=np.float32)])

    tcp_rel_b = transform_to_tcp_frame(obj_pos_b, ee_pos_b, ee_quat_b)
    assert tcp_rel_b.shape == (2, 3)
    assert np.allclose(tcp_rel_b[0], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(tcp_rel_b[1], [0.0, 0.0, 0.5], atol=1e-6)


def test_dataset_tcp_frame_toggle(dataset_path: Path) -> None:
    """Verify dataset loads in TCP-frame mode and records metadata correctly."""
    ds_world = LeRobotParquetDataset(
        dataset_path, horizon=1, include_ee_pose=True, include_object_pose=True, use_tcp_frame=False
    )
    ds_tcp = LeRobotParquetDataset(
        dataset_path, horizon=1, include_ee_pose=True, include_object_pose=True, use_tcp_frame=True
    )

    assert ds_world.obs_dim == 30
    assert ds_tcp.obs_dim == 30
    assert ds_world.get_stats_dict()["use_tcp_frame"] is False
    assert ds_tcp.get_stats_dict()["use_tcp_frame"] is True

    # Check that sample 0 differs in the relative position components (indices 26..29)
    obs_w, _ = ds_world[0]
    obs_t, _ = ds_tcp[0]
    # Indices: q(0..5), v(6..11), ee_pos(12..14), ee_quat(15..18),
    # obj_pos(19..21), obj_quat(22..25), rel_pos(26..28), grip(29)
    assert torch.allclose(obs_w[:26], obs_t[:26])
    assert torch.allclose(obs_w[29:], obs_t[29:])


def test_policy_to_policy_fn_tcp_frame_branching() -> None:
    """Verify runtime policy_fn respects use_tcp_frame flag in stats dictionary."""
    model = BehaviorCloningMLP(obs_dim=30, action_dim=7, horizon=1)
    stats_world = {
        "q_mean": [0.0] * 6,
        "q_std": [1.0] * 6,
        "v_mean": [0.0] * 6,
        "v_std": [1.0] * 6,
        "act_mean": [0.0] * 7,
        "act_std": [1.0] * 7,
        "use_tcp_frame": False,
    }
    stats_tcp = dict(stats_world)
    stats_tcp["use_tcp_frame"] = True

    fn_world = model.to_policy_fn(stats_world, device="cpu")
    fn_tcp = model.to_policy_fn(stats_tcp, device="cpu")

    js = JointState(
        positions_rad=(0.0,) * 6,
        velocities_rad_s=(0.0,) * 6,
        torques_nm=(0.0,) * 6,
    )
    # EE rotated +90 deg yaw
    w = float(np.cos(np.pi / 4.0))
    z = float(np.sin(np.pi / 4.0))
    ee_pose = SE3Pose(position=(0.0, 0.0, 0.0), quaternion_wxyz=(w, 0.0, 0.0, z))
    obj_pose = SE3Pose(position=(0.0, 1.0, 0.0), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))

    # In world frame: rel_pos = [0, 1, 0]
    # In TCP frame: rel_pos = [1, 0, 0]
    # Since weights are random, different inputs yield different action chunks
    act_world = fn_world(js, 0, ee_pose=ee_pose, object_pose=obj_pose, gripper_state=0.0)
    act_tcp = fn_tcp(js, 0, ee_pose=ee_pose, object_pose=obj_pose, gripper_state=0.0)
    assert not np.allclose(act_world, act_tcp)


def test_cli_train_policy_with_tcp_frame(
    dataset_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI train-policy accepts --tcp-frame and saves use_tcp_frame in metadata."""
    ckpt_path = tmp_path / "tcp_model.pt"

    rc = main(
        [
            "train-policy",
            "--dataset-dir",
            str(dataset_path),
            "--epochs",
            "1",
            "--batch-size",
            "128",
            "--horizon",
            "1",
            "--output-model",
            str(ckpt_path),
            "--tcp-frame",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    # Load checkpoint and assert metadata
    _, meta = BehaviorCloningMLP.load_checkpoint(ckpt_path, device="cpu")
    assert meta.get("use_tcp_frame") is True

