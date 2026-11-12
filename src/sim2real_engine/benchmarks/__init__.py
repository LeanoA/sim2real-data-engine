"""Cross-engine Sim-to-Sim evaluation and trajectory divergence benchmarks."""

from __future__ import annotations

from sim2real_engine.benchmarks.ood_dr_eval import (
    OODBenchmarkReport,
    OODRegimeMetrics,
    PerceptionNoiseConfig,
    PerceptionNoisePoint,
    PerceptionSensitivityCurve,
    evaluate_ood_robustness,
    run_perception_sensitivity_sweep,
)
from sim2real_engine.benchmarks.sim2sim_eval import (
    Sim2SimMetrics,
    create_simulator_backend,
    evaluate_sim2sim_gap,
)

__all__ = [
    "OODBenchmarkReport",
    "OODRegimeMetrics",
    "PerceptionNoiseConfig",
    "PerceptionNoisePoint",
    "PerceptionSensitivityCurve",
    "Sim2SimMetrics",
    "create_simulator_backend",
    "evaluate_ood_robustness",
    "evaluate_sim2sim_gap",
    "run_perception_sensitivity_sweep",
]
