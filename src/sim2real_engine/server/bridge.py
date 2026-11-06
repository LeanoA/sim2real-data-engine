"""Decoupled policy inference bridge and real-time latency telemetry."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.core.contracts import JointState
from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer

PolicyFn = Callable[[JointState, int], NDArray[np.float64]]


class LatencySummary(BaseModel):
    """Telemetry metrics summary for policy inference bridging latency distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_ms: float = Field(..., ge=0.0, description="Mean inference latency in milliseconds.")
    p50_ms: float = Field(..., ge=0.0, description="Median (P50) latency in milliseconds.")
    p95_ms: float = Field(..., ge=0.0, description="95th percentile latency in milliseconds.")
    p99_ms: float = Field(..., ge=0.0, description="99th percentile latency in milliseconds.")
    max_ms: float = Field(..., ge=0.0, description="Maximum observed latency in milliseconds.")
    sample_count: int = Field(..., ge=0, description="Total number of recorded inference steps.")


class PolicyInferenceBridge:
    """Bridges high-level policy inference chunks to high-frequency physics execution."""

    def __init__(
        self,
        buffer: TemporalEnsembleBuffer,
        simulated_latency_ms: float = 8.0,
        policy_fn: PolicyFn | None = None,
        replan_interval: int = 1,
    ) -> None:
        self.buffer = buffer
        self.simulated_latency_ms = max(0.0, float(simulated_latency_ms))
        self.replan_interval = max(1, int(replan_interval))
        self.policy_fn = policy_fn
        self._latencies_ms: list[float] = []

    def _default_synthetic_policy(
        self,
        observation: JointState,
        step: int,
    ) -> NDArray[np.float64]:
        """Generate a smooth target action chunk around current joint configuration."""
        q = np.asarray(observation.positions_rad, dtype=np.float64)
        h = self.buffer.horizon
        dof = self.buffer.action_dim
        chunk = np.zeros((h, dof), dtype=np.float64)
        for i in range(h):
            phase = (step + i) * 0.05
            chunk[i] = q + 0.015 * np.sin(phase + np.arange(dof) * 0.4)
        return chunk

    def infer_step(self, observation: JointState, step: int) -> NDArray[np.float64]:
        """Query policy, feed temporal ensemble buffer, record latency, and emit action."""
        t_start = time.perf_counter()

        if step % self.replan_interval == 0:
            if self.policy_fn is not None:
                chunk = self.policy_fn(observation, step)
            else:
                chunk = self._default_synthetic_policy(observation, step)
            self.buffer.push_chunk(chunk, step=step)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_latency = elapsed_ms + self.simulated_latency_ms
        self._latencies_ms.append(float(total_latency))

        return self.buffer.get_ensembled_action(current_step=step)

    def get_latency_metrics(self) -> LatencySummary:
        """Compute latency percentile distribution across recorded execution steps."""
        if not self._latencies_ms:
            return LatencySummary(
                mean_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
                sample_count=0,
            )

        arr = np.asarray(self._latencies_ms, dtype=np.float64)
        return LatencySummary(
            mean_ms=round(float(np.mean(arr)), 3),
            p50_ms=round(float(np.percentile(arr, 50)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            p99_ms=round(float(np.percentile(arr, 99)), 3),
            max_ms=round(float(np.max(arr)), 3),
            sample_count=len(self._latencies_ms),
        )

    def reset(self) -> None:
        """Reset internal buffer state and latency tracker history."""
        self.buffer.reset()
        self._latencies_ms.clear()
