"""Verification suite for Phase 4: LeRobotDataset v2 Parquet exporter & SHA-256 manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import EpisodeStepRecord, JointState, SE3Pose
from sim2real_engine.dataset.exporter import LeRobotEpisodeExporter
from sim2real_engine.dataset.manifest import DatasetManifest, FeatureStats


def _make_dummy_step(episode_id: int, step_index: int) -> EpisodeStepRecord:
    dof = 6
    positions = tuple(0.1 * i + 0.01 * step_index for i in range(dof))
    velocities = tuple(0.05 * i for i in range(dof))
    torques = tuple(0.01 * i for i in range(dof))
    targets = tuple(0.1 * i + 0.01 * (step_index + 1) for i in range(dof))

    return EpisodeStepRecord(
        episode_id=episode_id,
        step_index=step_index,
        timestamp_s=step_index * 0.02,
        joint_state=JointState(
            positions_rad=positions,
            velocities_rad_s=velocities,
            torques_nm=torques,
        ),
        end_effector_pose=SE3Pose(
            position=(0.3, 0.0, 0.2),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        privileged_object_pose=SE3Pose(
            position=(0.4, 0.0, 0.05),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        action_joint_targets_rad=targets,
        is_terminal=(step_index == 4),
    )


def test_feature_stats_contract_validation() -> None:
    """FeatureStats enforces vector dimension consistency and min <= max."""
    valid = FeatureStats(
        mean=(0.0, 1.0),
        std=(0.5, 0.2),
        min=(-1.0, 0.5),
        max=(1.0, 1.5),
    )
    assert len(valid.mean) == 2

    with pytest.raises(ValueError, match="share identical dimension"):
        FeatureStats(mean=(0.0, 1.0), std=(0.5,), min=(-1.0, 0.5), max=(1.0, 1.5))

    with pytest.raises(ValueError, match="must be non-negative"):
        FeatureStats(mean=(0.0, 1.0), std=(-0.1, 0.2), min=(-1.0, 0.5), max=(1.0, 1.5))

    with pytest.raises(ValueError, match="exceeds maximum"):
        FeatureStats(mean=(0.0, 1.0), std=(0.5, 0.2), min=(2.0, 0.5), max=(1.0, 1.5))


def test_exporter_parquet_round_trip(tmp_path: Path) -> None:
    """Exported episode must be a valid Parquet table matching written steps."""
    exporter = LeRobotEpisodeExporter(root_dir=tmp_path)
    steps = [_make_dummy_step(0, i) for i in range(5)]

    file_path = exporter.write_episode(episode_id=0, steps=steps)
    assert file_path.is_file()
    assert file_path.name == "episode_000000.parquet"

    table = pq.read_table(file_path)
    assert table.num_rows == 5
    assert "observation.state" in table.column_names
    assert "observation.object_position" in table.column_names
    assert "observation.object_quaternion" in table.column_names
    assert "observation.gripper_state" in table.column_names
    assert "action" in table.column_names
    assert "is_terminal" in table.column_names


def test_exporter_finalize_manifest_and_stats(tmp_path: Path) -> None:
    """Finalized manifest records correct step count, SHA-256 hashes, and normalization stats."""
    exporter = LeRobotEpisodeExporter(root_dir=tmp_path, dataset_name="test_robotics")
    ep0_steps = [_make_dummy_step(0, i) for i in range(4)]
    ep1_steps = [_make_dummy_step(1, i) for i in range(6)]

    exporter.write_episode(episode_id=0, steps=ep0_steps)
    exporter.write_episode(episode_id=1, steps=ep1_steps)

    manifest = exporter.finalize_manifest()
    assert manifest.dataset_name == "test_robotics"
    assert manifest.format_version == "lerobot_v2"
    assert manifest.num_episodes == 2
    assert manifest.total_steps == 10
    assert len(manifest.episode_sha256) == 2
    assert "data/episode_000000.parquet" in manifest.episode_sha256
    assert "data/episode_000001.parquet" in manifest.episode_sha256

    state_stats = manifest.feature_stats["observation.state"]
    assert len(state_stats.mean) == 6
    assert len(state_stats.std) == 6

    manifest_on_disk = tmp_path / "meta" / "manifest.json"
    assert manifest_on_disk.is_file()
    assert "observation.object_position" in manifest.feature_stats
    assert "observation.gripper_state" in manifest.feature_stats


def test_exporter_7d_reactive_action_export(tmp_path: Path) -> None:
    """Exporter properly serializes 7-DOF reactive actions and statistics."""
    exporter = LeRobotEpisodeExporter(root_dir=tmp_path, dataset_name="test_7d")
    steps: list[EpisodeStepRecord] = []
    for i in range(5):
        s = _make_dummy_step(0, i)
        s7 = EpisodeStepRecord(
            episode_id=s.episode_id,
            step_index=s.step_index,
            timestamp_s=s.timestamp_s,
            joint_state=s.joint_state,
            end_effector_pose=s.end_effector_pose,
            privileged_object_pose=s.privileged_object_pose,
            action_joint_targets_rad=(*s.action_joint_targets_rad, 1.0 if i >= 2 else 0.0),
            is_terminal=s.is_terminal,
        )
        steps.append(s7)

    exporter.write_episode(episode_id=0, steps=steps)
    manifest = exporter.finalize_manifest()

    action_stats = manifest.feature_stats["action"]
    assert len(action_stats.mean) == 7
    assert len(action_stats.std) == 7
    assert action_stats.max[6] == 1.0
    assert action_stats.min[6] == 0.0
    manifest_on_disk = tmp_path / "meta" / "manifest.json"
    raw_json = manifest_on_disk.read_text(encoding="utf-8")
    loaded_manifest = DatasetManifest.model_validate_json(raw_json)
    assert loaded_manifest == manifest


def test_exporter_sha256_tamper_detection(tmp_path: Path) -> None:
    """`verify_integrity` succeeds on clean datasets and catches file tampering."""
    exporter = LeRobotEpisodeExporter(root_dir=tmp_path)
    steps = [_make_dummy_step(0, i) for i in range(3)]
    ep_file = exporter.write_episode(episode_id=0, steps=steps)
    exporter.finalize_manifest()

    assert exporter.verify_integrity() is True

    # Tamper with the parquet file by modifying content
    raw_bytes = ep_file.read_bytes()
    ep_file.write_bytes(raw_bytes + b"\x00")
    assert exporter.verify_integrity() is False


def test_exporter_empty_inputs_raise(tmp_path: Path) -> None:
    """Exporter enforces non-empty step collections."""
    exporter = LeRobotEpisodeExporter(root_dir=tmp_path)
    with pytest.raises(ValueError, match="must contain at least one step"):
        exporter.write_episode(0, [])

    with pytest.raises(ValueError, match="no episode steps have been written"):
        exporter.finalize_manifest()


def test_cli_export_dataset_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI export-dataset outputs valid JSON envelope and passes integrity verification."""
    out_dir = str(tmp_path / "cli_dataset")
    ret = main(["export-dataset", "--episodes", "2", "--output-dir", out_dir, "--json"])
    assert ret == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["integrity_verified"] is True
    assert payload["manifest"]["num_episodes"] == 2
    assert payload["manifest"]["total_steps"] > 0


def test_cli_export_dataset_plain_and_invalid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI export-dataset without --json prints summary, and rejects invalid episodes."""
    out_dir = str(tmp_path / "cli_plain")
    ret = main(["export-dataset", "--episodes", "1", "--output-dir", out_dir])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Successfully exported 1 episodes" in captured.out
    assert "SHA-256 envelope integrity verified: True" in captured.out

    ret_err = main(["export-dataset", "--episodes", "0"])
    assert ret_err == 1
    captured_err = capsys.readouterr()
    assert "Number of episodes must be >= 1" in captured_err.err
