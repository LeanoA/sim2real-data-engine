"""Statistical inference, uncertainty quantification, and psychometric curve fitting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BinomialCI(BaseModel):
    """Binomial success rate with Wilson score confidence interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate: float = Field(..., ge=0.0, le=1.0, description="Observed proportion k / n.")
    ci_low: float = Field(..., ge=0.0, le=1.0, description="Lower bound of Wilson score CI.")
    ci_high: float = Field(..., ge=0.0, le=1.0, description="Upper bound of Wilson score CI.")
    successes: int = Field(..., ge=0, description="Observed number of successes (k).")
    trials: int = Field(..., ge=1, description="Total number of independent trials (n).")
    confidence: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Nominal confidence level."
    )


class DistributionStats(BaseModel):
    """Non-parametric and parametric summary statistics for continuous metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int = Field(..., ge=1, description="Sample count.")
    mean: float = Field(..., description="Sample arithmetic mean.")
    std: float = Field(..., ge=0.0, description="Sample standard deviation (ddof=1).")
    median: float = Field(..., description="Sample 50th percentile (median).")
    q25: float = Field(..., description="Sample 25th percentile.")
    q75: float = Field(..., description="Sample 75th percentile.")
    iqr: float = Field(..., ge=0.0, description="Interquartile range (Q75 - Q25).")
    ci95_low: float = Field(..., description="Lower 95% bootstrap percentile CI of the mean.")
    ci95_high: float = Field(..., description="Upper 95% bootstrap percentile CI of the mean.")
    min_val: float = Field(..., description="Sample minimum.")
    max_val: float = Field(..., description="Sample maximum.")


class PsychometricFit(BaseModel):
    """Binomial MLE logistic psychometric curve fit for noise tolerance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    converged: bool = Field(
        ..., description="True if MLE converged to a strictly degrading logistic curve (k > 0)."
    )
    sigma_50_mm: float | None = Field(
        default=None, description="Estimated noise level (mm) where success probability equals 50%."
    )
    sigma_90_mm: float | None = Field(
        default=None, description="Estimated noise level (mm) where success probability equals 90%."
    )
    sigma_10_mm: float | None = Field(
        default=None, description="Estimated noise level (mm) where success probability equals 10%."
    )
    transition_width_mm: float | None = Field(
        default=None,
        ge=0.0,
        description="90%-to-10% transition bandwidth (sigma_10 - sigma_90) in mm.",
    )
    slope_per_mm: float = Field(
        default=0.0, ge=0.0, description="Logistic degradation steepness parameter k (1/mm)."
    )
    intercept: float = Field(default=0.0, description="Log-odds intercept beta_0 at sigma=0.")
    pseudo_r2: float = Field(
        default=0.0, ge=0.0, le=1.0, description="McFadden's pseudo-R^2 against null intercept."
    )


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
    decimals: int = 4,
) -> BinomialCI:
    """Compute exact Wilson score confidence interval for a binomial proportion.

    Unlike the normal Wald approximation, Wilson score intervals maintain non-zero
    width and proper coverage at 0% and 100% empirical rates and small sample sizes.
    """
    if trials < 1:
        msg = f"trials must be >= 1; got {trials}"
        raise ValueError(msg)
    if successes < 0 or successes > trials:
        msg = f"successes must be in [0, {trials}]; got {successes}"
        raise ValueError(msg)
    if not (0.0 < confidence < 1.0):
        msg = f"confidence must be in (0, 1); got {confidence}"
        raise ValueError(msg)

    p_hat = successes / trials
    z = NormalDist().inv_cdf(0.5 + 0.5 * confidence)
    z2 = z * z
    denom = 1.0 + (z2 / trials)
    center = (p_hat + (z2 / (2.0 * trials))) / denom
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / trials) + (z2 / (4.0 * trials * trials)))
        / denom
    )

    rate_r = round(p_hat, decimals)
    low_r = round(max(0.0, min(1.0, center - margin)), decimals)
    high_r = round(max(0.0, min(1.0, center + margin)), decimals)

    # Guard against 4-decimal rounding flipping boundary inclusion
    low_r = min(low_r, rate_r)
    high_r = max(high_r, rate_r)

    return BinomialCI(
        rate=rate_r,
        ci_low=low_r,
        ci_high=high_r,
        successes=successes,
        trials=trials,
        confidence=confidence,
    )


def summarize_distribution(
    values: Sequence[float] | Sequence[int] | np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    decimals: int = 4,
) -> DistributionStats:
    """Compute parametric, robust non-parametric, and bootstrap CI statistics."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        msg = "Cannot summarize an empty sequence of values."
        raise ValueError(msg)

    n = int(arr.size)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    q25_val, median_val, q75_val = (
        float(x) for x in np.percentile(arr, [25.0, 50.0, 75.0])
    )
    iqr_val = max(0.0, q75_val - q25_val)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if n >= 2 and n_bootstrap >= 10 and std_val > 0.0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot_means = np.mean(arr[idx], axis=1)
        alpha = 1.0 - confidence
        ci_low_val, ci_high_val = (
            float(x)
            for x in np.percentile(
                boot_means, [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)]
            )
        )
    else:
        ci_low_val, ci_high_val = mean_val, mean_val

    return DistributionStats(
        n=n,
        mean=round(mean_val, decimals),
        std=round(std_val, decimals),
        median=round(median_val, decimals),
        q25=round(q25_val, decimals),
        q75=round(q75_val, decimals),
        iqr=round(iqr_val, decimals),
        ci95_low=round(ci_low_val, decimals),
        ci95_high=round(ci_high_val, decimals),
        min_val=round(min_val, decimals),
        max_val=round(max_val, decimals),
    )


def _binomial_log_likelihood(
    logits: np.ndarray, k: np.ndarray, n: np.ndarray
) -> float:
    """Compute numerically stable binomial log-likelihood up to constant combinatorial term."""
    # k * logit - n * log(1 + exp(logit))
    return float(np.sum(k * logits - n * np.logaddexp(0.0, logits)))


def fit_psychometric_logistic(
    sigmas_mm: Sequence[float],
    successes: Sequence[int],
    trials: Sequence[int],
    *,
    ridge_l2: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> PsychometricFit:
    """Fit a logistic psychometric curve via regularized Binomial MLE (IRLS / Newton-Raphson).

    Models P(success | sigma) = sigmoid(beta_0 + beta_1 * sigma).
    When beta_1 < 0, extracts analytical sigma_50, sigma_90, sigma_10, and slope k = -beta_1.
    """
    if not (len(sigmas_mm) == len(successes) == len(trials)):
        msg = "sigmas_mm, successes, and trials must have identical lengths."
        raise ValueError(msg)
    if len(sigmas_mm) < 2:
        return PsychometricFit(converged=False)

    x_raw = np.asarray(sigmas_mm, dtype=np.float64)
    k = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)

    if np.any(n < 1) or np.any(k < 0) or np.any(k > n):
        msg = "Invalid binomial counts: require trials >= 1 and 0 <= successes <= trials."
        raise ValueError(msg)

    x_mean = float(np.mean(x_raw))
    x_std = float(np.std(x_raw))
    if x_std < 1e-9:
        return PsychometricFit(converged=False)

    x = (x_raw - x_mean) / x_std
    design = np.column_stack([np.ones_like(x), x])

    # Initialize from empirical pooled log-odds
    total_k = float(np.sum(k))
    total_n = float(np.sum(n))
    p_pool = float(np.clip(total_k / total_n, 1e-4, 1.0 - 1e-4))
    beta = np.array([math.log(p_pool / (1.0 - p_pool)), 0.0], dtype=np.float64)

    penalty = np.diag([0.0, ridge_l2])
    converged = False

    def objective(b: np.ndarray) -> float:
        eta = np.clip(design @ b, -30.0, 30.0)
        ll = _binomial_log_likelihood(eta, k, n)
        return ll - 0.5 * float(b @ penalty @ b)

    cur_obj = objective(beta)

    for _ in range(max_iter):
        eta = np.clip(design @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = design.T @ (k - n * p) - (penalty @ beta)
        weights = n * p * (1.0 - p)
        hessian = (design.T * weights) @ design + penalty

        if float(np.linalg.cond(hessian)) > 1e12:
            break

        step = np.linalg.solve(hessian, grad)
        if float(np.linalg.norm(step)) < tol:
            converged = True
            break

        # Damped backtracking line search
        step_size = 1.0
        accepted = False
        for _ in range(12):
            candidate = beta + step_size * step
            cand_obj = objective(candidate)
            if cand_obj >= cur_obj - 1e-12:
                beta = candidate
                cur_obj = cand_obj
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            break

    # Transform parameters back to unstandardized sigma_mm
    beta_1 = float(beta[1] / x_std)
    beta_0 = float(beta[0] - beta_1 * x_mean)

    # McFadden's pseudo-R^2 on unpenalized log-likelihood
    eta_fit = np.clip(beta_0 + beta_1 * x_raw, -30.0, 30.0)
    ll_fit = _binomial_log_likelihood(eta_fit, k, n)
    if 0.0 < total_k < total_n:
        ll_null = total_k * math.log(p_pool) + (total_n - total_k) * math.log(1.0 - p_pool)
        pseudo_r2 = float(np.clip(1.0 - (ll_fit / ll_null), 0.0, 1.0)) if ll_null < -1e-9 else 0.0
    else:
        pseudo_r2 = 0.0

    # Strictly degrading curve requires negative slope beta_1 < 0
    if beta_1 >= -1e-5:
        return PsychometricFit(
            converged=False,
            slope_per_mm=0.0,
            intercept=round(beta_0, 4),
            pseudo_r2=round(pseudo_r2, 4),
        )

    slope_k = -beta_1
    sigma_50 = -beta_0 / beta_1
    ln9 = math.log(9.0)
    sigma_90 = sigma_50 - (ln9 / slope_k)
    sigma_10 = sigma_50 + (ln9 / slope_k)
    width = sigma_10 - sigma_90

    return PsychometricFit(
        converged=converged or (pseudo_r2 > 0.0),
        sigma_50_mm=round(sigma_50, 3),
        sigma_90_mm=round(sigma_90, 3),
        sigma_10_mm=round(sigma_10, 3),
        transition_width_mm=round(max(0.0, width), 3),
        slope_per_mm=round(slope_k, 4),
        intercept=round(beta_0, 4),
        pseudo_r2=round(pseudo_r2, 4),
    )
