"""Unit tests for statistical inference, distribution summaries, and psychometric fits."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim2real_engine.benchmarks.statistics import (
    fit_psychometric_logistic,
    summarize_distribution,
    wilson_score_interval,
)


def test_wilson_score_interval_bounds_and_edge_cases() -> None:
    """Wilson intervals retain non-zero width at 0% and 100% and match exact formula."""
    # 8 / 10 at 95% confidence -> ~[0.4902, 0.9433]
    ci_80 = wilson_score_interval(8, 10, confidence=0.95)
    assert ci_80.rate == pytest.approx(0.8, abs=1e-4)
    assert ci_80.ci_low == pytest.approx(0.4902, abs=1e-3)
    assert ci_80.ci_high == pytest.approx(0.9433, abs=1e-3)
    assert ci_80.successes == 8
    assert ci_80.trials == 10

    # Zero successes: upper bound must stay strictly positive
    ci_0 = wilson_score_interval(0, 10)
    assert ci_0.rate == 0.0
    assert ci_0.ci_low == 0.0
    assert 0.2 < ci_0.ci_high < 0.35

    # 100% successes: lower bound must stay strictly below 1.0
    ci_100 = wilson_score_interval(10, 10)
    assert ci_100.rate == 1.0
    assert 0.65 < ci_100.ci_low < 0.8
    assert ci_100.ci_high == 1.0

    # Single trial
    ci_1 = wilson_score_interval(1, 1)
    assert ci_1.ci_low < 1.0
    assert ci_1.ci_high == 1.0


def test_wilson_score_interval_validation() -> None:
    """Invalid counts or confidence levels raise ValueError."""
    with pytest.raises(ValueError, match="trials must be >= 1"):
        wilson_score_interval(0, 0)
    with pytest.raises(ValueError, match="successes must be in"):
        wilson_score_interval(11, 10)
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_score_interval(5, 10, confidence=1.0)


def test_summarize_distribution_parametric_and_robust_stats() -> None:
    """DistributionStats computes exact mean, sample SD, median, IQR, and deterministic CI."""
    vals = [1.0, 2.0, 3.0, 4.0, 10.0]
    stats = summarize_distribution(vals, n_bootstrap=500, seed=7)
    assert stats.n == 5
    assert stats.mean == pytest.approx(4.0, abs=1e-4)
    assert stats.std == pytest.approx(float(np.std(vals, ddof=1)), abs=1e-4)
    assert stats.median == pytest.approx(3.0, abs=1e-4)
    assert stats.q25 == pytest.approx(2.0, abs=1e-4)
    assert stats.q75 == pytest.approx(4.0, abs=1e-4)
    assert stats.iqr == pytest.approx(2.0, abs=1e-4)
    assert stats.min_val == 1.0
    assert stats.max_val == 10.0
    assert stats.ci95_low <= stats.mean <= stats.ci95_high

    # Determinism check
    stats_repeat = summarize_distribution(vals, n_bootstrap=500, seed=7)
    assert stats == stats_repeat

    # Single-element edge case
    single = summarize_distribution([5.5])
    assert single.n == 1
    assert single.std == 0.0
    assert single.iqr == 0.0
    assert single.ci95_low == single.ci95_high == 5.5

    with pytest.raises(ValueError, match="empty"):
        summarize_distribution([])


def test_fit_psychometric_logistic_recovers_sigma_50_and_slope() -> None:
    """Binomial MLE recovers planted sigma_50 and slope on synthetic sweep data."""
    sigmas = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    true_s50 = 10.0
    true_k = 0.6
    n_trials = 200
    probs = [1.0 / (1.0 + math.exp(true_k * (s - true_s50))) for s in sigmas]
    successes = [int(round(p * n_trials)) for p in probs]

    fit = fit_psychometric_logistic(sigmas, successes, [n_trials] * len(sigmas))
    assert fit.converged is True
    assert fit.sigma_50_mm is not None
    assert fit.sigma_50_mm == pytest.approx(true_s50, abs=0.25)
    assert fit.slope_per_mm == pytest.approx(true_k, abs=0.05)
    assert fit.sigma_90_mm is not None and fit.sigma_10_mm is not None
    assert fit.sigma_90_mm < fit.sigma_50_mm < fit.sigma_10_mm
    assert fit.transition_width_mm == pytest.approx(
        fit.sigma_10_mm - fit.sigma_90_mm, abs=1e-2
    )
    assert fit.pseudo_r2 > 0.5


def test_fit_psychometric_logistic_separation_and_flat_cases() -> None:
    """Handles complete separation gracefully and marks flat/increasing data non-degrading."""
    # Complete separation: 100% up to 5mm, 0% from 10mm onward
    sep_fit = fit_psychometric_logistic(
        sigmas_mm=[0.0, 5.0, 10.0, 15.0],
        successes=[10, 10, 0, 0],
        trials=[10, 10, 10, 10],
    )
    assert sep_fit.converged is True
    assert sep_fit.sigma_50_mm is not None
    assert 5.0 <= sep_fit.sigma_50_mm <= 10.0

    # Flat constant curve -> cannot identify degradation threshold
    flat_fit = fit_psychometric_logistic(
        sigmas_mm=[0.0, 5.0, 10.0],
        successes=[5, 5, 5],
        trials=[10, 10, 10],
    )
    assert flat_fit.converged is False
    assert flat_fit.sigma_50_mm is None
    assert flat_fit.slope_per_mm == 0.0
