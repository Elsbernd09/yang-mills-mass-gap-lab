"""
Monte Carlo diagnostics for SU(2) lattice gauge theory.

This module provides tools for:
1. Burn-in removal.
2. Autocorrelation estimation.
3. Integrated autocorrelation time.
4. Effective sample size.

These diagnostics make the simulation methodology more credible and honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AutocorrelationDiagnostics:
    """Summary of autocorrelation diagnostics for a time series."""

    autocorrelation: np.ndarray
    integrated_autocorrelation_time: float
    effective_sample_size: float


def remove_burn_in(values: np.ndarray, burn_in: int) -> np.ndarray:
    """
    Remove the first `burn_in` samples from a one-dimensional time series.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative.")

    if burn_in >= len(values):
        raise ValueError("burn_in must be smaller than the number of samples.")

    return values[burn_in:]


def autocorrelation_function(values: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Estimate the normalized autocorrelation function of a one-dimensional series.

    The lag-0 value is normalized to 1 for nonconstant data.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    n = len(values)

    if n < 2:
        raise ValueError("Need at least two samples.")

    if max_lag is None:
        max_lag = n - 1

    if max_lag < 0 or max_lag >= n:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(values).")

    centered = values - np.mean(values)
    variance = np.mean(centered * centered)

    if np.isclose(variance, 0.0):
        result = np.zeros(max_lag + 1, dtype=float)
        result[0] = 1.0
        return result

    correlations = np.empty(max_lag + 1, dtype=float)

    for lag in range(max_lag + 1):
        left = centered[: n - lag]
        right = centered[lag:]
        correlations[lag] = np.mean(left * right) / variance

    return correlations


def integrated_autocorrelation_time(
    autocorrelation: np.ndarray,
    cutoff: int | None = None,
) -> float:
    """
    Estimate integrated autocorrelation time.

    A simple estimator is

        tau_int = 1/2 + sum_{t=1}^{cutoff} rho(t)

    Negative tail values can make this unstable, so by default this function
    stops at the first nonpositive autocorrelation after lag 0.
    """
    autocorrelation = np.asarray(autocorrelation, dtype=float)

    if autocorrelation.ndim != 1:
        raise ValueError("autocorrelation must be one-dimensional.")

    if len(autocorrelation) < 2:
        raise ValueError("Need at least two autocorrelation values.")

    if cutoff is None:
        positive_lags = []

        for lag in range(1, len(autocorrelation)):
            if autocorrelation[lag] <= 0:
                break
            positive_lags.append(lag)

        if len(positive_lags) == 0:
            return 0.5

        cutoff = positive_lags[-1]

    if cutoff < 1 or cutoff >= len(autocorrelation):
        raise ValueError("cutoff must satisfy 1 <= cutoff < len(autocorrelation).")

    tau = 0.5 + float(np.sum(autocorrelation[1 : cutoff + 1]))

    return max(tau, 0.5)


def effective_sample_size(n_samples: int, tau_int: float) -> float:
    """
    Estimate effective sample size from integrated autocorrelation time.

    For correlated Markov chain samples, a rough estimate is

        N_eff = N / (2 tau_int).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if tau_int <= 0:
        raise ValueError("tau_int must be positive.")

    return float(n_samples / (2.0 * tau_int))


def diagnose_autocorrelation(
    values: np.ndarray,
    burn_in: int = 0,
    max_lag: int | None = None,
) -> AutocorrelationDiagnostics:
    """
    Remove burn-in and compute autocorrelation diagnostics.
    """
    cleaned = remove_burn_in(values, burn_in=burn_in)

    if max_lag is None:
        max_lag = min(len(cleaned) - 1, max(1, len(cleaned) // 2))

    autocorr = autocorrelation_function(cleaned, max_lag=max_lag)
    tau_int = integrated_autocorrelation_time(autocorr)
    ess = effective_sample_size(len(cleaned), tau_int)

    return AutocorrelationDiagnostics(
        autocorrelation=autocorr,
        integrated_autocorrelation_time=tau_int,
        effective_sample_size=ess,
    )
