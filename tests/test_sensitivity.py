"""Tests for 2^4 orthogonal ANOVA sensitivity analysis and True OOD benchmarks."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from sim2real_engine.benchmarks.sensitivity import (
    MultivariateSensitivityReport,
    compute_orthogonal_2k_anova,
    run_multivariate_sensitivity,
)
from sim2real_engine.cli import main


def test_compute_orthogonal_2k_anova_exact_partition_and_effects() -> None:
    """Balanced 2^4 factorial ANOVA recovers planted effects and partitions variance to 1.0."""
    cells = list(itertools.product((-1.0, 1.0), repeat=4))
    coded_rows: list[tuple[float, float, float, float]] = []
    cell_ids: list[int] = []
    response: list[float] = []

    for cell_id, (x0, x1, x2, x3) in enumerate(cells):
        for rep_noise in (-0.5, 0.5):
            coded_rows.append((x0, x1, x2, x3))
            cell_ids.append(cell_id)
            y = 10.0 + (3.0 * x0) - (1.0 * x1) + (2.0 * x0 * x1) + rep_noise
            response.append(y)

    res = compute_orthogonal_2k_anova(
        coded_factors=np.asarray(coded_rows, dtype=np.float64),
        cell_ids=cell_ids,
        response=response,
    )

    assert res.main_deltas == pytest.approx([6.0, -2.0, 0.0, 0.0], abs=1e-4)
    # Pair (0, 1) is index 0 in combinations(range(4), 2)
    assert res.interaction_pairs[0] == (0, 1)
    assert res.interaction_deltas[0] == pytest.approx(4.0, abs=1e-4)
    assert all(d == pytest.approx(0.0, abs=1e-4) for d in res.interaction_deltas[1:])
    assert res.higher_order_eta_sq == pytest.approx(0.0, abs=1e-6)

    # Exact orthogonal ANOVA partition: sum of all eta^2 terms == 1.0
    total_eta_sq = (
        sum(res.main_eta_sq)
        + sum(res.interaction_eta_sq)
        + res.higher_order_eta_sq
        + res.residual_eta_sq
    )
    assert total_eta_sq == pytest.approx(1.0, abs=1e-4)
    assert res.main_f_stats[0] is not None and res.main_f_stats[0] > 100.0
    assert res.main_f_stats[2] == pytest.approx(0.0, abs=1e-6)


def test_run_multivariate_sensitivity_execution_and_oat() -> None:
    """2^4 factorial study executes with Common Random Numbers and optional OAT sweeps."""
    report = run_multivariate_sensitivity(
        model_path="checkpoints/bc_pick_place.pt",
        backend_name="mujoco_native",
        episodes_per_cell=2,
        base_seed=42,
        max_steps=10,
        include_oat=False,
    )
    assert isinstance(report, MultivariateSensitivityReport)
    assert report.total_episodes == 32
    assert report.overall_grasp_ci.trials == 32
    assert report.overall_chained_ci.trials == 32
    assert len(report.main_effects) == 4
    assert len(report.interactions_2way) == 6
    factor_names = {eff.factor_name for eff in report.main_effects}
    assert factor_names == {"mass_scale", "friction_scale", "damping_scale", "spatial_shift_m"}
    assert report.dominant_grasp_bottleneck in factor_names
    assert report.dominant_task_bottleneck in factor_names

    # OAT smoke check
    report_oat = run_multivariate_sensitivity(
        model_path="checkpoints/bc_pick_place.pt",
        backend_name="mujoco_native",
        episodes_per_cell=1,
        base_seed=42,
        max_steps=8,
        include_oat=True,
    )
    assert len(report_oat.oat_sweeps) == 4
    assert len(report_oat.oat_sweeps[0].points) == 5


def test_cli_sensitivity_analysis_json_and_plain(capsys: pytest.CaptureFixture[str]) -> None:
    """`sim2real sensitivity-analysis` produces valid JSON and ANOVA summary table."""
    rc_json = main([
        "sensitivity-analysis",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--episodes-per-cell", "1",
        "--max-steps", "8",
        "--json",
    ])
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["design"] == "2^4_full_factorial"
    assert payload["total_episodes"] == 16
    assert len(payload["main_effects"]) == 4

    rc_plain = main([
        "sensitivity-analysis",
        "--model-path", "checkpoints/bc_pick_place.pt",
        "--episodes-per-cell", "1",
        "--max-steps", "8",
    ])
    assert rc_plain == 0
    out_plain = capsys.readouterr().out
    assert "Multivariate Sensitivity Analysis" in out_plain
    assert "Dominant Bottlenecks" in out_plain
