"""
Effective mass estimation from correlation functions.

For a correlation function with approximate exponential decay,

    C(t) ~ A exp(-m t),

one can estimate an effective mass using

    m_eff(t) = log(C(t) / C(t+1)).

This is a numerical diagnostic, not a proof of a positive Yang-Mills mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class EffectiveMassResult:
    """Container for effective mass estimates."""

    lags: np.ndarray
    effective_masses: np.ndarray


@dataclass
class ExponentialFitResult:
    """Container for exponential fit results."""

    amplitude: float
    mass: float
    covariance: np.ndarray


def effective_mass(correlation: np.ndarray) -> EffectiveMassResult:
    """
    Compute effective masses from a correlation function.

    Only consecutive positive correlation values are used.
    """
    correlation = np.asarray(correlation, dtype=float)

    if correlation.ndim != 1:
        raise ValueError("Correlation must be one-dimensional.")

    if len(correlation) < 2:
        raise ValueError("Need at least two correlation values.")

    masses = []
    lags = []

    for t in range(len(correlation) - 1):
        c_t = correlation[t]
        c_next = correlation[t + 1]

        if c_t > 0 and c_next > 0:
            masses.append(float(np.log(c_t / c_next)))
            lags.append(t)

    return EffectiveMassResult(
        lags=np.asarray(lags, dtype=int),
        effective_masses=np.asarray(masses, dtype=float),
    )


def exponential_decay(t: np.ndarray, amplitude: float, mass: float) -> np.ndarray:
    """Model C(t) = amplitude * exp(-mass * t)."""
    return amplitude * np.exp(-mass * t)


def fit_exponential_mass(
    correlation: np.ndarray,
    min_lag: int = 0,
    max_lag: int | None = None,
) -> ExponentialFitResult:
    """
    Fit positive correlation data to C(t) = A exp(-m t).

    Parameters
    ----------
    correlation:
        One-dimensional correlation values.
    min_lag:
        First lag included in the fit.
    max_lag:
        Last lag included in the fit. If None, use the end.

    Returns
    -------
    ExponentialFitResult
        Estimated amplitude, mass, and covariance matrix.
    """
    correlation = np.asarray(correlation, dtype=float)

    if correlation.ndim != 1:
        raise ValueError("Correlation must be one-dimensional.")

    if max_lag is None:
        max_lag = len(correlation) - 1

    if min_lag < 0 or max_lag <= min_lag:
        raise ValueError("Invalid lag range.")

    t_values = np.arange(len(correlation))
    mask = (
        (t_values >= min_lag)
        & (t_values <= max_lag)
        & (correlation > 0)
        & np.isfinite(correlation)
    )

    fit_t = t_values[mask]
    fit_c = correlation[mask]

    if len(fit_t) < 2:
        raise ValueError("Need at least two positive points for exponential fit.")

    initial_amplitude = float(fit_c[0])
    initial_mass = 0.1

    params, covariance = curve_fit(
        exponential_decay,
        fit_t,
        fit_c,
        p0=[initial_amplitude, initial_mass],
        maxfev=10000,
    )

    amplitude, mass = params

    return ExponentialFitResult(
        amplitude=float(amplitude),
        mass=float(mass),
        covariance=covariance,
    )
