"""Mathematical and control-theoretic tests for Temporal Ensembling properties."""

from __future__ import annotations

import math
import numpy as np
import pytest

from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer


def test_convex_combination_unity_sum() -> None:
    """Temporal ensembling weights must mathematically form a convex combination summing to 1.0."""
    horizon = 16
    decay_rate = 0.05
    buffer = TemporalEnsembleBuffer(action_dim=6, horizon=horizon, decay_rate=decay_rate)

    # Push full set of H overlapping chunks for step = horizon - 1
    target_step = 15
    for t0 in range(target_step + 1):
        dummy_chunk = np.ones((horizon, 6), dtype=np.float64) * float(t0)
        buffer.push_chunk(dummy_chunk, step=t0)

    # Compute manual weights: w_k = exp(-m * k)
    weights = [math.exp(-decay_rate * (target_step - t0)) for t0 in range(target_step + 1)]
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]

    assert math.isclose(sum(normalized_weights), 1.0, rel_tol=1e-9)

    # Action output must match the analytical weighted sum
    ensembled = buffer.get_ensembled_action(current_step=target_step)
    expected_val = sum(w * float(t0) for w, t0 in zip(normalized_weights, range(target_step + 1)))
    np.testing.assert_allclose(
        ensembled,
        expected_val,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Ensembled action did not match exact convex combination.",
    )


def test_monotonic_exponential_weight_decay() -> None:
    """When decay_rate > 0, newer action predictions must strictly outweigh older predictions."""
    decay_rate = 0.1
    horizon = 8
    buffer = TemporalEnsembleBuffer(action_dim=4, horizon=horizon, decay_rate=decay_rate)

    # Verify decay ordering for prediction ages k = 0, 1, 2, ...
    weights = [math.exp(-decay_rate * k) for k in range(horizon)]
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1], (
            f"Weight monotonicity violated at age {i}: w_{i}={weights[i]} <= w_{i+1}={weights[i+1]}"
        )


def test_flat_weighting_when_decay_is_zero() -> None:
    """When decay_rate == 0.0, temporal ensembling acts as a uniform moving average."""
    horizon = 4
    buffer = TemporalEnsembleBuffer(action_dim=2, horizon=horizon, decay_rate=0.0)

    # Push 4 chunks with values 1.0, 2.0, 3.0, 4.0
    for t0 in range(4):
        chunk = np.full((horizon, 2), float(t0 + 1), dtype=np.float64)
        buffer.push_chunk(chunk, step=t0)

    # At step 3, average of [1, 2, 3, 4] is 2.5
    action = buffer.get_ensembled_action(current_step=3)
    np.testing.assert_allclose(action, 2.5, rtol=1e-6)


def test_action_jerk_and_chatter_attenuation() -> None:
    """Temporal ensembling must provably attenuate high-frequency control chatter."""
    rng = np.random.default_rng(12345)
    horizon = 8
    decay_rate = 0.08
    dim = 6
    steps = 60

    buffer = TemporalEnsembleBuffer(action_dim=dim, horizon=horizon, decay_rate=decay_rate)

    raw_actions: list[np.ndarray] = []
    ensembled_actions: list[np.ndarray] = []

    # Generate synthetic smooth trajectory + high-frequency noise
    for t in range(steps):
        t_sec = t * 0.02
        smooth_center = np.sin(t_sec * 2.0 + np.arange(dim))
        # Add random prediction jitter across horizon
        noise = rng.normal(0.0, 0.05, size=(horizon, dim))
        chunk = smooth_center[np.newaxis, :] + noise

        buffer.push_chunk(chunk, step=t)
        ens_act = buffer.get_ensembled_action(current_step=t)

        raw_actions.append(chunk[0])
        ensembled_actions.append(ens_act)

    # Compute control jerk (mean L2 norm of successive action differences)
    raw_diffs = np.linalg.norm(np.diff(raw_actions, axis=0), axis=-1)
    ens_diffs = np.linalg.norm(np.diff(ensembled_actions, axis=0), axis=-1)

    raw_jerk = float(np.mean(raw_diffs))
    ens_jerk = float(np.mean(ens_diffs))

    # Ensembling must attenuate chatter by at least 25%
    jerk_reduction_pct = (raw_jerk - ens_jerk) / raw_jerk * 100.0
    assert jerk_reduction_pct > 25.0, (
        f"Expected >25% jerk reduction; got {jerk_reduction_pct:.2f}% (raw={raw_jerk:.4f}, ens={ens_jerk:.4f})"
    )


def test_buffer_pruning_memory_leak_immunity() -> None:
    """Expired chunks must be actively pruned to prevent memory growth over long rollouts."""
    horizon = 16
    buffer = TemporalEnsembleBuffer(action_dim=7, horizon=horizon, decay_rate=0.05)

    # Simulate 5,000 steps of receding-horizon pushing
    for step in range(5000):
        chunk = np.ones((horizon, 7), dtype=np.float64)
        buffer.push_chunk(chunk, step=step)
        _ = buffer.get_ensembled_action(current_step=step)

        # Buffer must never hold more than horizon active chunks
        assert buffer.active_chunks_count <= horizon, (
            f"Buffer leaked memory at step {step}: {buffer.active_chunks_count} chunks retained (limit {horizon})"
        )


def test_unensembled_mode_returns_latest_chunk_slice() -> None:
    """When enabled=False, buffer returns exact slice from latest chunk without blending."""
    horizon = 4
    buffer = TemporalEnsembleBuffer(action_dim=3, horizon=horizon, enabled=False)

    chunk_0 = np.array([[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13]], dtype=np.float64)
    buffer.push_chunk(chunk_0, step=0)

    # Step 0 gets chunk_0[0]
    act_0 = buffer.get_ensembled_action(current_step=0)
    np.testing.assert_allclose(act_0, [10, 10, 10])

    # Step 1 without new push gets chunk_0[1]
    act_1 = buffer.get_ensembled_action(current_step=1)
    np.testing.assert_allclose(act_1, [11, 11, 11])
