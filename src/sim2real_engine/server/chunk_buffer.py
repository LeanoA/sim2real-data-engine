"""Receding-horizon action chunk ring buffer with exponential temporal ensembling."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


class TemporalEnsembleBuffer:
    """Ring buffer for multi-step action predictions with exponential temporal decay ensembling.

    Implements the temporal ensembling algorithm used by foundation visuomotor policies
    (ACT, Diffusion Policy, OpenPI):
        u_t = sum(exp(-m * k) * a_{t | t-k}) / sum(exp(-m * k))
    where k = t - t_0 is the prediction age and m is the decay rate.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int = 16,
        decay_rate: float = 0.05,
    ) -> None:
        if action_dim < 1:
            msg = f"action_dim must be >= 1; got {action_dim}"
            raise ValueError(msg)
        if horizon < 1:
            msg = f"horizon must be >= 1; got {horizon}"
            raise ValueError(msg)
        if decay_rate < 0.0:
            msg = f"decay_rate must be non-negative; got {decay_rate}"
            raise ValueError(msg)

        self.action_dim = action_dim
        self.horizon = horizon
        self.decay_rate = decay_rate
        # Key: base_step t_0 -> Value: array of shape (horizon, action_dim)
        self._chunks: dict[int, NDArray[np.float64]] = {}

    def push_chunk(self, chunk: NDArray[np.float64], step: int) -> None:
        """Store an action chunk predicted at `step` covering steps [step, step + horizon - 1]."""
        if step < 0:
            msg = f"step must be non-negative; got {step}"
            raise ValueError(msg)

        arr = np.asarray(chunk, dtype=np.float64)
        if arr.shape != (self.horizon, self.action_dim):
            msg = f"Expected chunk shape ({self.horizon}, {self.action_dim}); got {arr.shape}"
            raise ValueError(msg)

        self._chunks[step] = arr

        # Prune fully expired chunks: if t_0 + horizon <= step, it won't cover future steps
        expired_keys = [t0 for t0 in self._chunks if t0 + self.horizon <= step]
        for t0 in expired_keys:
            del self._chunks[t0]

    def get_ensembled_action(self, current_step: int) -> NDArray[np.float64]:
        """Compute the exponentially weighted temporal ensemble action for `current_step`."""
        if current_step < 0:
            msg = f"current_step must be non-negative; got {current_step}"
            raise ValueError(msg)

        weights: list[float] = []
        actions: list[NDArray[np.float64]] = []

        for t0, chunk in self._chunks.items():
            if t0 <= current_step < t0 + self.horizon:
                k = current_step - t0
                w = math.exp(-self.decay_rate * k)
                offset = current_step - t0
                weights.append(w)
                actions.append(chunk[offset])

        if not weights:
            msg = f"No active action chunks available covering step {current_step}."
            raise RuntimeError(msg)

        weight_arr = np.array(weights, dtype=np.float64)
        action_arr = np.array(actions, dtype=np.float64)
        total_weight = float(np.sum(weight_arr))

        weighted_sum = np.sum(weight_arr[:, np.newaxis] * action_arr, axis=0)
        ensembled: NDArray[np.float64] = weighted_sum / total_weight
        return ensembled

    def reset(self) -> None:
        """Clear all active action chunks from buffer."""
        self._chunks.clear()

    @property
    def active_chunks_count(self) -> int:
        """Return the number of active chunks currently retained in buffer."""
        return len(self._chunks)
