"""Cross-engine Sim-to-Sim evaluation and trajectory divergence benchmarks."""

from __future__ import annotations

from sim2real_engine.benchmarks.control_diagnostics import (
    ControlDiagnosticsSummary,
    RegimeControlDiagnostics,
    aggregate_control_diagnostics,
    compute_episode_control_diagnostics,
    compute_fft_spectral_metrics,
    compute_physical_joint_jerk,
)
from sim2real_engine.benchmarks.ood_dr_eval import (
    OODBenchmarkReport,
    OODRegimeMetrics,
    PerceptionNoiseConfig,
    PerceptionNoisePoint,
    PerceptionSensitivityCurve,
    build_default_ood_config,
    build_in_distribution_dr_config,
    evaluate_ood_robustness,
    run_perception_sensitivity_sweep,
)
from sim2real_engine.benchmarks.sensitivity import (
    DEFAULT_FACTORS,
    FactorEffectSummary,
    FactorLevelSpec,
    FactorOATSweep,
    InteractionEffectSummary,
    MultivariateSensitivityReport,
    OATSweepPoint,
    OrthogonalAnovaResult,
    compute_orthogonal_2k_anova,
    run_multivariate_sensitivity,
)
from sim2real_engine.benchmarks.sim2sim_eval import (
    Sim2SimMetrics,
    create_simulator_backend,
    evaluate_sim2sim_gap,
)
from sim2real_engine.benchmarks.statistics import (
    BinomialCI,
    DistributionStats,
    PsychometricFit,
    fit_psychometric_logistic,
    summarize_distribution,
    wilson_score_interval,
)

__all__ = [
    "BinomialCI",
    "ControlDiagnosticsSummary",
    "DEFAULT_FACTORS",
    "DistributionStats",
    "FactorEffectSummary",
    "FactorLevelSpec",
    "FactorOATSweep",
    "InteractionEffectSummary",
    "MultivariateSensitivityReport",
    "OATSweepPoint",
    "OODBenchmarkReport",
    "OODRegimeMetrics",
    "OrthogonalAnovaResult",
    "PerceptionNoiseConfig",
    "PerceptionNoisePoint",
    "PerceptionSensitivityCurve",
    "PsychometricFit",
    "RegimeControlDiagnostics",
    "Sim2SimMetrics",
    "aggregate_control_diagnostics",
    "build_default_ood_config",
    "build_in_distribution_dr_config",
    "compute_episode_control_diagnostics",
    "compute_fft_spectral_metrics",
    "compute_orthogonal_2k_anova",
    "compute_physical_joint_jerk",
    "create_simulator_backend",
    "evaluate_ood_robustness",
    "evaluate_sim2sim_gap",
    "fit_psychometric_logistic",
    "run_multivariate_sensitivity",
    "run_perception_sensitivity_sweep",
    "summarize_distribution",
    "wilson_score_interval",
]
