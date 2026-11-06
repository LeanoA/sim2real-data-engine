"""Verification suite for Phase 5: Decoupled policy inference bridge & temporal ensembling."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim2real_engine.cli import main
from sim2real_engine.core.contracts import JointState
from sim2real_engine.server.bridge import LatencySummary, PolicyInferenceBridge
from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer


def test_chunk_buffer_validation() -> None:
    """TemporalEnsembleBuffer rejects invalid dimensions and mismatched chunk arrays."""
    with pytest.raises(ValueError, match="action_dim must be >= 1"):
        TemporalEnsembleBuffer(action_dim=0)

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        TemporalEnsembleBuffer(action_dim=6, horizon=0)

    with pytest.raises(ValueError, match="decay_rate must be non-negative"):
        TemporalEnsembleBuffer(action_dim=6, decay_rate=-0.1)

    buf = TemporalEnsembleBuffer(action_dim=3, horizon=4)
    with pytest.raises(ValueError, match="Expected chunk shape"):
        buf.push_chunk(np.zeros((5, 3)), step=0)

    with pytest.raises(ValueError, match="step must be non-negative"):
        buf.push_chunk(np.zeros((4, 3)), step=-1)

    with pytest.raises(ValueError, match="current_step must be non-negative"):
        buf.get_ensembled_action(current_step=-1)


def test_temporal_ensemble_exponential_math() -> None:
    """Overlapping chunks must follow the exact exp(-m * k) weighted formula."""
    decay = 0.2
    buf = TemporalEnsembleBuffer(action_dim=1, horizon=3, decay_rate=decay)

    # Chunk 0: predicts for steps [0, 1, 2] with constant value 10.0
    buf.push_chunk(np.full((3, 1), 10.0), step=0)
    # Chunk 1: predicts for steps [1, 2, 3] with constant value 20.0
    buf.push_chunk(np.full((3, 1), 20.0), step=1)

    # At step 1:
    # Chunk 0 has age k=1 (weight exp(-0.2*1)), value 10.0
    # Chunk 1 has age k=0 (weight exp(-0.2*0) = 1.0), value 20.0
    w0 = math.exp(-decay * 1.0)
    w1 = 1.0
    expected = (10.0 * w0 + 20.0 * w1) / (w0 + w1)

    ensembled = buf.get_ensembled_action(current_step=1)
    assert ensembled.shape == (1,)
    assert math.isclose(float(ensembled[0]), expected, rel_tol=1e-7)


def test_temporal_ensemble_zero_decay_is_simple_average() -> None:
    """When decay_rate=0.0, temporal ensembling equals uniform arithmetic mean."""
    buf = TemporalEnsembleBuffer(action_dim=2, horizon=4, decay_rate=0.0)

    chunk_a = np.ones((4, 2)) * 2.0
    chunk_b = np.ones((4, 2)) * 6.0

    buf.push_chunk(chunk_a, step=0)
    buf.push_chunk(chunk_b, step=1)

    # At step 1, both chunks overlap: average of 2.0 and 6.0 is 4.0
    action = buf.get_ensembled_action(current_step=1)
    np.testing.assert_allclose(action, [4.0, 4.0], atol=1e-12)


def test_chunk_pruning_and_missing_coverage_error() -> None:
    """Expired chunks are pruned and querying uncovered steps raises RuntimeError."""
    buf = TemporalEnsembleBuffer(action_dim=2, horizon=2)
    buf.push_chunk(np.zeros((2, 2)), step=0)  # covers 0, 1
    assert buf.active_chunks_count == 1

    # Pushing at step 2 expires chunk at step 0 (0 + 2 <= 2)
    buf.push_chunk(np.ones((2, 2)), step=2)  # covers 2, 3
    assert buf.active_chunks_count == 1

    with pytest.raises(RuntimeError, match="No active action chunks available"):
        buf.get_ensembled_action(current_step=0)

    # Step 2 is covered
    act = buf.get_ensembled_action(current_step=2)
    np.testing.assert_allclose(act, [1.0, 1.0], atol=1e-12)


def test_policy_inference_bridge_execution_and_telemetry() -> None:
    """PolicyInferenceBridge steps through control loop and records P50/P95/P99 latency."""
    buffer = TemporalEnsembleBuffer(action_dim=6, horizon=8, decay_rate=0.05)
    bridge = PolicyInferenceBridge(
        buffer=buffer,
        simulated_latency_ms=5.0,
        replan_interval=2,
    )

    dummy_state = JointState(
        positions_rad=(0.0, 0.1, -0.1, 0.0, 0.2, 0.0),
        velocities_rad_s=(0.0,) * 6,
        torques_nm=(0.0,) * 6,
    )

    for step in range(20):
        action = bridge.infer_step(dummy_state, step=step)
        assert action.shape == (6,)

    metrics = bridge.get_latency_metrics()
    assert isinstance(metrics, LatencySummary)
    assert metrics.sample_count == 20
    assert metrics.mean_ms >= 5.0
    assert metrics.p50_ms >= 5.0
    assert metrics.max_ms >= metrics.p99_ms >= metrics.p95_ms >= metrics.p50_ms

    bridge.reset()
    assert bridge.get_latency_metrics().sample_count == 0


def test_cli_benchmark_bridge_json(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real benchmark-bridge --steps 20 --chunk-horizon 8 --json` returns valid JSON."""
    ret = main(["benchmark-bridge", "--steps", "20", "--chunk-horizon", "8", "--json"])
    assert ret == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["steps"] == 20
    assert payload["chunk_horizon"] == 8
    assert payload["metrics"]["sample_count"] == 20
    assert payload["metrics"]["p99_ms"] > 0.0


def test_cli_benchmark_bridge_plain_and_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real benchmark-bridge` prints formatted summary, rejects invalid steps."""
    ret = main(["benchmark-bridge", "--steps", "10"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Policy Bridge Benchmark (10 steps" in captured.out
    assert "Latency: Mean=" in captured.out

    ret_err = main(["benchmark-bridge", "--steps", "0"])
    assert ret_err == 1
    captured_err = capsys.readouterr()
    assert "Steps and horizon must be >= 1" in captured_err.err
