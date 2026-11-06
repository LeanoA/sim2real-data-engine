"""Cross-engine Sim-to-Sim evaluation and trajectory divergence benchmarks."""

from __future__ import annotations

from sim2real_engine.benchmarks.sim2sim_eval import (
    Sim2SimMetrics,
    create_simulator_backend,
    evaluate_sim2sim_gap,
)

__all__ = [
    "Sim2SimMetrics",
    "create_simulator_backend",
    "evaluate_sim2sim_gap",
]
