"""Multivariate OOD factor sensitivity and orthogonal 2^k ANOVA variance attribution."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sim2real_engine.benchmarks.ood_dr_eval import (
    PerceptionNoiseConfig,
    run_regime_episodes,
)
from sim2real_engine.benchmarks.statistics import (
    BinomialCI,
    DistributionStats,
    summarize_distribution,
    wilson_score_interval,
)
from sim2real_engine.core.contracts import BackendName, SE3Pose
from sim2real_engine.learning.policy import BehaviorCloningMLP
from sim2real_engine.randomization.samplers import EpisodePerturbation


class FactorLevelSpec(BaseModel):
    """Low/High experimental levels and OAT sweep grid for one physical/spatial factor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    low_level: float
    high_level: float
    oat_levels: tuple[float, ...]


DEFAULT_FACTORS: tuple[FactorLevelSpec, ...] = (
    FactorLevelSpec(
        name="mass_scale",
        low_level=0.85,
        high_level=2.00,
        oat_levels=(0.50, 1.00, 1.50, 2.00, 2.50),
    ),
    FactorLevelSpec(
        name="friction_scale",
        low_level=0.35,
        high_level=1.15,
        oat_levels=(0.25, 0.45, 0.70, 1.00, 1.30),
    ),
    FactorLevelSpec(
        name="damping_scale",
        low_level=0.90,
        high_level=1.50,
        oat_levels=(0.60, 0.90, 1.00, 1.30, 1.70),
    ),
    FactorLevelSpec(
        name="spatial_shift_m",
        low_level=0.00,
        high_level=0.07,
        oat_levels=(0.00, 0.03, 0.06, 0.09, 0.12),
    ),
)


class OrthogonalAnovaResult(BaseModel):
    """Exact orthogonal ANOVA variance decomposition for a balanced 2^k factorial design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    main_deltas: list[float]
    main_eta_sq: list[float]
    main_f_stats: list[float | None]
    interaction_pairs: list[tuple[int, int]]
    interaction_deltas: list[float]
    interaction_eta_sq: list[float]
    higher_order_eta_sq: float
    residual_eta_sq: float


class FactorEffectSummary(BaseModel):
    """Main-effect magnitude, 95% Wilson CIs, and ANOVA variance share for one factor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_name: str
    low_level: float
    high_level: float
    grasp_low_ci: BinomialCI
    grasp_high_ci: BinomialCI
    grasp_delta_pp: float = Field(
        ..., description="High-level minus Low-level grasp success in percentage points."
    )
    grasp_eta_sq: float = Field(
        ..., ge=0.0, le=1.0, description="ANOVA variance share (eta^2) for grasp success."
    )
    grasp_f_stat: float | None = None
    chained_low_ci: BinomialCI
    chained_high_ci: BinomialCI
    chained_delta_pp: float = Field(
        ..., description="High-level minus Low-level chained success in percentage points."
    )
    chained_eta_sq: float = Field(
        ..., ge=0.0, le=1.0, description="ANOVA variance share (eta^2) for chained success."
    )
    chained_f_stat: float | None = None
    reach_error_delta_mm: float = Field(
        ..., description="High-level minus Low-level mean min reach error (mm)."
    )
    reach_error_eta_sq: float = Field(..., ge=0.0, le=1.0)


class InteractionEffectSummary(BaseModel):
    """Pairwise factor interaction variance share and effect size."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_a: str
    factor_b: str
    grasp_interaction_pp: float
    grasp_eta_sq: float = Field(..., ge=0.0, le=1.0)
    chained_interaction_pp: float
    chained_eta_sq: float = Field(..., ge=0.0, le=1.0)
    reach_error_eta_sq: float = Field(..., ge=0.0, le=1.0)


class OATSweepPoint(BaseModel):
    """Single level evaluation in a One-Factor-at-a-Time (OAT) stress sweep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: float
    grasp_ci: BinomialCI
    chained_ci: BinomialCI
    reach_error_stats: DistributionStats


class FactorOATSweep(BaseModel):
    """One-Factor-at-a-Time stress curve for a single physical/spatial axis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_name: str
    points: list[OATSweepPoint]


class MultivariateSensitivityReport(BaseModel):
    """Full 2^4 factorial ANOVA sensitivity report with optional OAT stress curves."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str
    backend: BackendName
    design: str = "2^4_full_factorial"
    episodes_per_cell: int = Field(..., ge=1)
    total_episodes: int = Field(..., ge=16)
    overall_grasp_ci: BinomialCI
    overall_chained_ci: BinomialCI
    main_effects: list[FactorEffectSummary]
    interactions_2way: list[InteractionEffectSummary]
    higher_order_eta_sq_grasp: float = Field(..., ge=0.0, le=1.0)
    residual_eta_sq_grasp: float = Field(..., ge=0.0, le=1.0)
    higher_order_eta_sq_chained: float = Field(..., ge=0.0, le=1.0)
    residual_eta_sq_chained: float = Field(..., ge=0.0, le=1.0)
    dominant_grasp_bottleneck: str
    dominant_task_bottleneck: str
    oat_sweeps: list[FactorOATSweep] = Field(default_factory=list)


def compute_orthogonal_2k_anova(
    coded_factors: np.ndarray,
    cell_ids: Sequence[int] | np.ndarray,
    response: Sequence[float] | np.ndarray,
) -> OrthogonalAnovaResult:
    """Decompose total response variance across orthogonal 2^k contrasts and within-cell error."""
    x = np.asarray(coded_factors, dtype=np.float64)
    cells = np.asarray(cell_ids, dtype=np.int64)
    y = np.asarray(response, dtype=np.float64).ravel()

    if x.ndim != 2 or x.shape[0] != y.size or cells.size != y.size:
        msg = "coded_factors, cell_ids, and response must share sample length N."
        raise ValueError(msg)

    n_samples, k_factors = x.shape
    unique_cells = np.unique(cells)
    num_cells = unique_cells.size

    y_mean = float(np.mean(y))
    ss_total = float(np.sum((y - y_mean) ** 2))

    # Within-cell pure replication residual sum of squares
    ss_residual = 0.0
    for cell_id in unique_cells:
        mask = cells == cell_id
        y_cell = y[mask]
        cell_mean = float(np.mean(y_cell))
        ss_residual += float(np.sum((y_cell - cell_mean) ** 2))

    df_residual = max(0, n_samples - num_cells)
    ms_residual = (ss_residual / df_residual) if df_residual > 0 else 0.0
    denom_total = ss_total if ss_total > 1e-12 else 0.0

    main_deltas: list[float] = []
    main_eta_sq: list[float] = []
    main_f_stats: list[float | None] = []

    for j in range(k_factors):
        col = x[:, j]
        ss_j = float(np.sum(col * y) ** 2) / n_samples
        eta_j = (ss_j / denom_total) if denom_total > 0.0 else 0.0
        delta_j = float((2.0 / n_samples) * np.sum(col * y))
        f_j = round(ss_j / ms_residual, 3) if ms_residual > 1e-12 else None

        main_deltas.append(round(delta_j, 4))
        main_eta_sq.append(round(float(np.clip(eta_j, 0.0, 1.0)), 6))
        main_f_stats.append(f_j)

    interaction_pairs: list[tuple[int, int]] = []
    interaction_deltas: list[float] = []
    interaction_eta_sq: list[float] = []

    for a, b in itertools.combinations(range(k_factors), 2):
        c_ab = x[:, a] * x[:, b]
        ss_ab = float(np.sum(c_ab * y) ** 2) / n_samples
        eta_ab = (ss_ab / denom_total) if denom_total > 0.0 else 0.0
        delta_ab = float((2.0 / n_samples) * np.sum(c_ab * y))

        interaction_pairs.append((a, b))
        interaction_deltas.append(round(delta_ab, 4))
        interaction_eta_sq.append(round(float(np.clip(eta_ab, 0.0, 1.0)), 6))

    # 3-way and higher interactions
    ss_higher = 0.0
    for order in range(3, k_factors + 1):
        for combo in itertools.combinations(range(k_factors), order):
            contrast = np.prod(x[:, list(combo)], axis=1)
            ss_higher += float(np.sum(contrast * y) ** 2) / n_samples

    higher_eta = (ss_higher / denom_total) if denom_total > 0.0 else 0.0
    residual_eta = (ss_residual / denom_total) if denom_total > 0.0 else 0.0

    return OrthogonalAnovaResult(
        main_deltas=main_deltas,
        main_eta_sq=main_eta_sq,
        main_f_stats=main_f_stats,
        interaction_pairs=interaction_pairs,
        interaction_deltas=interaction_deltas,
        interaction_eta_sq=interaction_eta_sq,
        higher_order_eta_sq=round(float(np.clip(higher_eta, 0.0, 1.0)), 6),
        residual_eta_sq=round(float(np.clip(residual_eta, 0.0, 1.0)), 6),
    )


def _build_perturbation(
    episode_id: int,
    mass_scale: float,
    friction_scale: float,
    damping_scale: float,
    spatial_shift_m: float,
) -> EpisodePerturbation:
    return EpisodePerturbation(
        episode_id=episode_id,
        derived_seed=episode_id,
        mass_scale=mass_scale,
        friction_scale=friction_scale,
        damping_scale=damping_scale,
        lighting_intensity=1.0,
        camera_pose_offset=SE3Pose(),
        object_x_offset_m=0.0,
        object_y_offset_m=spatial_shift_m,
    )


def run_multivariate_sensitivity(
    model_path: str | Path,
    backend_name: BackendName = "mujoco_native",
    episodes_per_cell: int = 3,
    base_seed: int = 42,
    max_steps: int = 120,
    factors: tuple[FactorLevelSpec, ...] = DEFAULT_FACTORS,
    include_oat: bool = False,
    enable_ensembling: bool = True,
    perception_noise: PerceptionNoiseConfig | None = None,
) -> MultivariateSensitivityReport:
    """Run balanced 2^k factorial sensitivity study with orthogonal ANOVA attribution."""
    if episodes_per_cell < 1:
        msg = f"episodes_per_cell must be >= 1; got {episodes_per_cell}"
        raise ValueError(msg)
    if len(factors) != 4:
        msg = f"Expected 4 factor specifications; got {len(factors)}"
        raise ValueError(msg)

    ckpt = Path(model_path)
    if not ckpt.is_file():
        msg = f"Policy checkpoint not found: {ckpt}"
        raise FileNotFoundError(msg)

    model, metadata = BehaviorCloningMLP.load_checkpoint(ckpt)

    design_cells = list(itertools.product((-1.0, 1.0), repeat=4))
    perturbations: list[EpisodePerturbation] = []
    episode_seeds: list[int] = []
    coded_rows: list[tuple[float, float, float, float]] = []
    cell_ids: list[int] = []

    run_idx = 0
    for cell_id, (c_mass, c_fric, c_damp, c_shift) in enumerate(design_cells):
        mass_val = factors[0].high_level if c_mass > 0 else factors[0].low_level
        fric_val = factors[1].high_level if c_fric > 0 else factors[1].low_level
        damp_val = factors[2].high_level if c_damp > 0 else factors[2].low_level
        shift_val = factors[3].high_level if c_shift > 0 else factors[3].low_level

        for rep in range(episodes_per_cell):
            perturbations.append(
                _build_perturbation(run_idx, mass_val, fric_val, damp_val, shift_val)
            )
            # Common random numbers across all 16 factorial cells for variance reduction
            episode_seeds.append(base_seed + rep)
            coded_rows.append((c_mass, c_fric, c_damp, c_shift))
            cell_ids.append(cell_id)
            run_idx += 1

    _, outcomes = run_regime_episodes(
        model=model,
        metadata=metadata,
        backend_name=backend_name,
        num_episodes=len(perturbations),
        base_seed=base_seed,
        max_steps=max_steps,
        perturbations=perturbations,
        enable_ensembling=enable_ensembling,
        regime_name="factorial_2p4_sensitivity",
        perception_noise=perception_noise,
        episode_seeds=episode_seeds,
    )

    coded_arr = np.asarray(coded_rows, dtype=np.float64)
    cell_arr = np.asarray(cell_ids, dtype=np.int64)
    y_grasp = np.asarray([1.0 if o.grasp_ok else 0.0 for o in outcomes], dtype=np.float64)
    y_chained = np.asarray([1.0 if o.chained_ok else 0.0 for o in outcomes], dtype=np.float64)
    y_reach = np.asarray([o.min_reach_error_mm for o in outcomes], dtype=np.float64)

    anova_grasp = compute_orthogonal_2k_anova(coded_arr, cell_arr, y_grasp)
    anova_chained = compute_orthogonal_2k_anova(coded_arr, cell_arr, y_chained)
    anova_reach = compute_orthogonal_2k_anova(coded_arr, cell_arr, y_reach)

    main_effects: list[FactorEffectSummary] = []
    for idx, spec in enumerate(factors):
        low_mask = coded_arr[:, idx] < 0
        high_mask = coded_arr[:, idx] > 0
        n_half = int(np.sum(low_mask))

        g_low_k = int(np.sum(y_grasp[low_mask]))
        g_high_k = int(np.sum(y_grasp[high_mask]))
        c_low_k = int(np.sum(y_chained[low_mask]))
        c_high_k = int(np.sum(y_chained[high_mask]))

        main_effects.append(
            FactorEffectSummary(
                factor_name=spec.name,
                low_level=spec.low_level,
                high_level=spec.high_level,
                grasp_low_ci=wilson_score_interval(g_low_k, n_half),
                grasp_high_ci=wilson_score_interval(g_high_k, n_half),
                grasp_delta_pp=round(anova_grasp.main_deltas[idx] * 100.0, 2),
                grasp_eta_sq=anova_grasp.main_eta_sq[idx],
                grasp_f_stat=anova_grasp.main_f_stats[idx],
                chained_low_ci=wilson_score_interval(c_low_k, n_half),
                chained_high_ci=wilson_score_interval(c_high_k, n_half),
                chained_delta_pp=round(anova_chained.main_deltas[idx] * 100.0, 2),
                chained_eta_sq=anova_chained.main_eta_sq[idx],
                chained_f_stat=anova_chained.main_f_stats[idx],
                reach_error_delta_mm=round(anova_reach.main_deltas[idx], 3),
                reach_error_eta_sq=anova_reach.main_eta_sq[idx],
            )
        )

    main_effects_sorted = sorted(
        main_effects, key=lambda e: (e.chained_eta_sq, abs(e.chained_delta_pp)), reverse=True
    )

    interactions_2way: list[InteractionEffectSummary] = []
    for pair_idx, (a, b) in enumerate(anova_grasp.interaction_pairs):
        interactions_2way.append(
            InteractionEffectSummary(
                factor_a=factors[a].name,
                factor_b=factors[b].name,
                grasp_interaction_pp=round(anova_grasp.interaction_deltas[pair_idx] * 100.0, 2),
                grasp_eta_sq=anova_grasp.interaction_eta_sq[pair_idx],
                chained_interaction_pp=round(
                    anova_chained.interaction_deltas[pair_idx] * 100.0, 2
                ),
                chained_eta_sq=anova_chained.interaction_eta_sq[pair_idx],
                reach_error_eta_sq=anova_reach.interaction_eta_sq[pair_idx],
            )
        )

    interactions_sorted = sorted(
        interactions_2way, key=lambda it: (it.chained_eta_sq, it.grasp_eta_sq), reverse=True
    )

    dominant_grasp = max(main_effects, key=lambda e: (e.grasp_eta_sq, abs(e.grasp_delta_pp)))
    dominant_task = main_effects_sorted[0]

    oat_sweeps: list[FactorOATSweep] = []
    if include_oat:
        oat_eps = max(3, episodes_per_cell)
        for spec in factors:
            pts: list[OATSweepPoint] = []
            for lvl in spec.oat_levels:
                m_val = lvl if spec.name == "mass_scale" else 1.0
                f_val = lvl if spec.name == "friction_scale" else 1.0
                d_val = lvl if spec.name == "damping_scale" else 1.0
                s_val = lvl if spec.name == "spatial_shift_m" else 0.0
                perts = [_build_perturbation(i, m_val, f_val, d_val, s_val) for i in range(oat_eps)]
                _, oat_out = run_regime_episodes(
                    model=model,
                    metadata=metadata,
                    backend_name=backend_name,
                    num_episodes=oat_eps,
                    base_seed=base_seed,
                    max_steps=max_steps,
                    perturbations=perts,
                    enable_ensembling=enable_ensembling,
                    regime_name=f"oat_{spec.name}_{lvl}",
                    perception_noise=perception_noise,
                )
                g_k = sum(1 for o in oat_out if o.grasp_ok)
                c_k = sum(1 for o in oat_out if o.chained_ok)
                r_stats = summarize_distribution(
                    [o.min_reach_error_mm for o in oat_out], seed=base_seed, decimals=2
                )
                pts.append(
                    OATSweepPoint(
                        level=lvl,
                        grasp_ci=wilson_score_interval(g_k, oat_eps),
                        chained_ci=wilson_score_interval(c_k, oat_eps),
                        reach_error_stats=r_stats,
                    )
                )
            oat_sweeps.append(FactorOATSweep(factor_name=spec.name, points=pts))

    total_n = len(outcomes)
    return MultivariateSensitivityReport(
        model_path=str(ckpt),
        backend=backend_name,
        episodes_per_cell=episodes_per_cell,
        total_episodes=total_n,
        overall_grasp_ci=wilson_score_interval(int(np.sum(y_grasp)), total_n),
        overall_chained_ci=wilson_score_interval(int(np.sum(y_chained)), total_n),
        main_effects=main_effects_sorted,
        interactions_2way=interactions_sorted,
        higher_order_eta_sq_grasp=anova_grasp.higher_order_eta_sq,
        residual_eta_sq_grasp=anova_grasp.residual_eta_sq,
        higher_order_eta_sq_chained=anova_chained.higher_order_eta_sq,
        residual_eta_sq_chained=anova_chained.residual_eta_sq,
        dominant_grasp_bottleneck=dominant_grasp.factor_name,
        dominant_task_bottleneck=dominant_task.factor_name,
        oat_sweeps=oat_sweeps,
    )
